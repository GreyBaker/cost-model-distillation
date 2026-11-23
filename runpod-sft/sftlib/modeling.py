from __future__ import annotations

import os
import warnings


def cuda_device(value, for_training=True):
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use one process and one GPU, not torchrun/DDP")
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Native-BF16 CUDA GPU required; no CPU/FP16 fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if for_training and properties.total_memory / 1024**3 < value["gpu_memory_warning_gib"]:
        warnings.warn(f"{properties.name}: below advisory {value['gpu_memory_warning_gib']} GiB. "
                      "The unchanged recipe may OOM; validate fit with saved pilots on this GPU.", RuntimeWarning, stacklevel=2)
    return device


def load(path, device, trainable, value):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if model_config.model_type != "t5" or not model_config.is_encoder_decoder:
        raise ValueError("Expected a T5 encoder-decoder checkpoint")
    if trainable:
        model_config.dropout_rate = value["dropout"]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, config=model_config, torch_dtype=torch.float32 if trainable else torch.bfloat16,
        device_map={"": str(device)}, low_cpu_mem_usage=True, weights_only=True,
        local_files_only=True, trust_remote_code=False)
    if trainable:
        model.train()
        model.config.use_cache = False
        if value["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.requires_grad_(False).eval()
    return tokenizer, model


def loss_sum(model, inputs, labels):
    """Teacher-forced CE sum; caller normalizes over the effective batch tokens."""
    import torch
    import torch.nn.functional as F
    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError("Empty reference sequence")
    shifted = model.prepare_decoder_input_ids_from_labels(labels=labels)
    mask = torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
    output = model(**inputs, decoder_input_ids=shifted, decoder_attention_mask=mask, use_cache=False)
    return F.cross_entropy(output.logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="sum")


def generate(model, tokenizer, inputs, limit):
    import torch
    from transformers import GenerationConfig
    settings = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1, min_length=0,
                                min_new_tokens=0, no_repeat_ngram_size=0, repetition_penalty=1.0,
                                length_penalty=1.0, eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                decoder_start_token_id=model.config.decoder_start_token_id, use_cache=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ids = model.generate(**inputs, generation_config=settings)
    if not ids[:, 0].eq(model.config.decoder_start_token_id).all():
        raise ValueError("Unexpected decoder start")
    return ids[:, 1:]
