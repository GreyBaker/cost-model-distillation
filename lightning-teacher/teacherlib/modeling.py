from __future__ import annotations

import os
import warnings


def cuda_device(config, for_training=True):
    import torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use a single process/GPU, not torchrun, DDP or a Lightning distributed launcher")
    if (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8
            or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("A native-BF16 CUDA GPU is required; no CPU/FP16 fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if for_training and properties.total_memory / 1024 ** 3 < config["gpu_memory_warning_gib"]:
        warnings.warn(
            f"{properties.name} reports {properties.total_memory / 1024 ** 3:.1f} GiB total VRAM, "
            f"below the advisory {config['gpu_memory_warning_gib']} GiB threshold. "
            "Training will attempt the unchanged recipe. Total VRAM is not free VRAM or a fit guarantee; "
            "run the training/save/resume pilot before committing to the full run.",
            RuntimeWarning, stacklevel=2)
    return device


def load(path, device, trainable, config):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if model_config.model_type != "t5" or not model_config.is_encoder_decoder:
        raise ValueError("Expected a T5 encoder-decoder checkpoint")
    if trainable:
        model_config.dropout_rate = config["dropout"]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, config=model_config, torch_dtype=torch.float32 if trainable else torch.bfloat16,
        device_map={"": str(device)}, low_cpu_mem_usage=True, weights_only=True,
        local_files_only=True, trust_remote_code=False)
    if trainable:
        model.train()
        model.config.use_cache = False
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.requires_grad_(False).eval()
    return tokenizer, model


def loss(model, inputs, labels):
    import torch
    import torch.nn.functional as F
    shifted = model.prepare_decoder_input_ids_from_labels(labels)
    valid = labels.ne(-100)
    mask = torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
    outputs = model(**inputs, decoder_input_ids=shifted, decoder_attention_mask=mask, use_cache=False)
    values = F.cross_entropy(outputs.logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    if not valid.any(dim=1).all():
        raise ValueError("Empty reference sequence")
    return (values.sum(dim=1) / valid.sum(dim=1)).mean()


def generate(model, tokenizer, inputs, limit):
    import torch
    from transformers import GenerationConfig
    settings = GenerationConfig(max_new_tokens=limit, do_sample=False, num_beams=1, min_length=0,
                                min_new_tokens=0, no_repeat_ngram_size=0, repetition_penalty=1.0,
                                length_penalty=1.0, eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                decoder_start_token_id=model.config.decoder_start_token_id, use_cache=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = model.generate(**inputs, generation_config=settings)
    if not tokens[:, 0].eq(model.config.decoder_start_token_id).all():
        raise ValueError("Unexpected decoder start")
    return tokens[:, 1:]
