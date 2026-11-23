from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path


def device_and_precision(name: str, precision: str):
    import torch
    device = torch.device(name)
    if precision not in ("bfloat16", "float32"):
        raise ValueError("Only bfloat16 or float32 supported. FP16/quantization are not silent fallbacks.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing an accidental CPU training run")
        torch.cuda.set_device(device)
        if precision == "bfloat16" and torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError("Native BF16 requires an appropriate CUDA GPU; select FP32 explicitly or change hardware")
    elif device.type != "cpu" or precision != "float32":
        raise ValueError("CPU execution requires explicitly selected float32")
    return device


def autocast(device, precision: str):
    import torch
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bfloat16" else nullcontext()


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The prototype requires explicit PAD and EOS tokens")
    return tokenizer


def load_model(path: Path, device, precision: str, trainable: bool, dropout: float = 0.0):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config.model_type != "t5" or not config.is_encoder_decoder:
        raise ValueError("This prototype is specifically for T5 encoder-decoder checkpoints")
    if trainable:
        config.dropout_rate = dropout
    # Keep trainable parameters/master optimizer weights FP32; autocast controls compute.
    dtype = torch.float32 if trainable or precision == "float32" else torch.bfloat16
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path, config=config, local_files_only=True, trust_remote_code=False,
        torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": str(device)},
        weights_only=True,
    )
    if trainable:
        model.train()
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def check_pair(student, teacher, student_tokenizer, teacher_tokenizer) -> None:
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Token mappings differ; naive tokenwise GKD would be invalid")
    if student_tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map:
        raise ValueError("Special token conventions differ")
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError("Model output dimensions differ")
    for text in ("summarize: A short article.", "The price was £3.50 — not $4.", "New York\nBBC News"):
        if student_tokenizer.encode(text) != teacher_tokenizer.encode(text):
            raise ValueError("Representative tokenizer encodings differ")


def encode_sources(tokenizer, rows: list[dict], prefix: str, limit: int, device, pad_to_max_length=False):
    return tokenizer([prefix + row["document"] for row in rows], return_tensors="pt",
                     padding="max_length" if pad_to_max_length else True,
                     truncation=True, max_length=limit).to(device)


def target_mask(ids, eos_token_id: int, pad_token_id: int):
    """Generation-position mask: keep through first EOS, ignore later padding.

    A PAD ID emitted before EOS is a real sampled token, not batch padding.
    For reference/cached padded tensors, additionally supply their known position
    mask to labels_from_ids. Token identity alone cannot distinguish these roles.
    """
    import torch
    eos = ids.eq(eos_token_id)
    before_or_first_eos = (eos.cumsum(dim=1) - eos.to(torch.long)).eq(0)
    return before_or_first_eos


def labels_from_ids(ids, eos_token_id: int, pad_token_id: int, attention_mask=None):
    mask = target_mask(ids, eos_token_id, pad_token_id)
    if attention_mask is not None:
        mask = mask & attention_mask.bool()
    if not mask.any(dim=1).all():
        raise ValueError("Encountered an empty target; refusing an undefined sequence loss")
    return ids.masked_fill(~mask, -100)


def reference_labels(tokenizer, rows, limit: int, device):
    encoded = tokenizer(text_target=[row["reference"] for row in rows], return_tensors="pt",
                        padding=True, truncation=True, max_length=limit).to(device)
    return labels_from_ids(encoded["input_ids"], tokenizer.eos_token_id, tokenizer.pad_token_id,
                           attention_mask=encoded["attention_mask"])


def decoder_inputs(model, labels):
    import torch
    # T5 already aligns logits[t] to labels[t] after this shift. No second shift.
    ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    valid = labels.ne(-100)
    mask = torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)
    return {"decoder_input_ids": ids, "decoder_attention_mask": mask}


def generate_targets(model, tokenizer, sources, limit: int, sample: bool, temperature: float = 1.0):
    import torch
    from transformers import GenerationConfig
    if limit < 1 or (sample and temperature <= 0):
        raise ValueError("Invalid generation length or temperature")
    kwargs = dict(max_new_tokens=limit, do_sample=sample, num_beams=1,
                  min_length=0, min_new_tokens=0, no_repeat_ngram_size=0,
                  repetition_penalty=1.0, length_penalty=1.0,
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                  decoder_start_token_id=model.config.decoder_start_token_id, use_cache=True)
    if sample:
        kwargs.update(temperature=temperature, top_k=0, top_p=1.0)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            ids = model.generate(**sources, generation_config=GenerationConfig(**kwargs))
    finally:
        model.train(was_training)
    if not ids[:, 0].eq(model.config.decoder_start_token_id).all():
        raise ValueError("Unexpected T5 generation prefix")
    return ids[:, 1:]


def divergence_loss(student_logits, teacher_logits, labels, beta: float):
    """Full-vocabulary GKD: mean valid tokens per sequence, then mean examples.

    Endpoints follow TRL: 0 = KL(T||S); 1 = KL(S||T). Interior is weighted JSD.
    No temperature softening: the prototype fixes both loss temperatures at 1.
    """
    import math
    import torch
    import torch.nn.functional as F
    if not 0 <= beta <= 1:
        raise ValueError("beta must be within [0, 1]")
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    if beta == 0:
        terms = t.exp() * (t - s)
    elif beta == 1:
        terms = s.exp() * (s - t)
    else:
        mixture = torch.logaddexp(s + math.log1p(-beta), t + math.log(beta))
        terms = beta * t.exp() * (t - mixture) + (1 - beta) * s.exp() * (s - mixture)
    mask = labels.ne(-100)
    lengths = mask.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("Zero-length target")
    token_loss = terms.sum(dim=-1)
    return (token_loss.masked_fill(~mask, 0).sum(dim=1) / lengths).mean()
