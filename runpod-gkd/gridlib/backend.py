"""ML implementation: imported only by explicit Pod worker commands."""
from __future__ import annotations

from pathlib import Path

from .policy import PROTOCOL, TRAIN_N, VALIDATION_N
from .shared import environment


def runtime(settings):
    import torch
    from sftlib.modeling import cuda_device
    device = cuda_device({"gpu_memory_warning_gib": 75})
    torch.cuda.set_per_process_memory_fraction(settings["allocator_fraction"], device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = torch.cuda.get_device_properties(device)
    signature = {**environment(), "gpu": info.name,
                 "total_memory": info.total_memory, "capability": list(torch.cuda.get_device_capability(device)),
                 "cuda_runtime": torch.version.cuda}
    return device, signature


def load_pair(bound, settings, student_path=None):
    from sftlib import modeling, data
    from sftlib.core import verify_tree, read_json
    device, signature = runtime(settings)
    paths = bound["inputs"]["paths"]
    source = student_path or paths["student_checkpoint"]
    # Verify bytes before safe deserialization, not just manifest identities.
    for directory in (source, paths["teacher_export"]):
        directory = Path(directory)
        verify_tree(directory, read_json(directory / "COMPLETE.json")["files"])
    value = {**PROTOCOL, "gradient_checkpointing": True}
    tokenizer, student = modeling.load(source, device, True, value)
    teacher_tokenizer, teacher = modeling.load(paths["teacher_export"], device, False, value)
    if (tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
            or tokenizer.special_tokens_map != teacher_tokenizer.special_tokens_map
            or student.config.vocab_size != teacher.config.vocab_size
            or student.config.decoder_start_token_id != teacher.config.decoder_start_token_id):
        raise ValueError("Student/teacher token mappings or output dimensions differ")
    dataset = data.load(Path(bound["inputs"]["sft_work"]) / "student")
    if len(dataset["train"]) != TRAIN_N or len(dataset["validation"]) != VALIDATION_N:
        raise ValueError("Use the complete official prepared train/validation pool")
    return device, signature, tokenizer, student, teacher, dataset


def optimizer(student, rate):
    from transformers import Adafactor
    return Adafactor(student.parameters(), lr=rate, eps=(1e-30, 1e-3), clip_threshold=1.0,
                     decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=False,
                     relative_step=False, warmup_init=False)


def generate(student, tokenizer, inputs, sample, force_cap=False):
    import torch
    from transformers import GenerationConfig
    options = dict(max_new_tokens=64, min_new_tokens=63 if force_cap else 0,
                   min_length=0, do_sample=sample, num_beams=1,
                   no_repeat_ngram_size=0, repetition_penalty=1.0, length_penalty=1.0,
                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                   decoder_start_token_id=student.config.decoder_start_token_id, use_cache=True)
    if sample:
        options.update(temperature=1.0, top_k=0, top_p=1.0)
    previous = student.training
    student.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = student.generate(**inputs, generation_config=GenerationConfig(**options))
    finally:
        student.train(previous)
    if not ids[:, 0].eq(student.config.decoder_start_token_id).all():
        raise ValueError("Unexpected T5 decoder prefix")
    return ids[:, 1:]


def generated_labels(ids, tokenizer):
    import torch
    # Preserve every position through first EOS, including sampled PAD before EOS.
    eos = ids.eq(tokenizer.eos_token_id)
    valid = (eos.cumsum(dim=1) - eos.long()).eq(0)
    labels = ids.masked_fill(~valid, -100)
    return torch.nn.functional.pad(labels, (0, 64 - labels.shape[1]), value=-100)


def forward_kl(student_logits, teacher_logits, labels):
    """Eq.2: sum of sequence token means (caller divides by batch examples)."""
    import torch
    import torch.nn.functional as F
    s = F.log_softmax(student_logits.float(), dim=-1)
    t = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    mask = labels.ne(-100)
    lengths = mask.sum(dim=1)
    if not lengths.gt(0).all():
        raise ValueError("Empty trajectory")
    per_token = (t.exp() * (t - s)).sum(dim=-1)
    return (per_token.masked_fill(~mask, 0).sum(dim=1) / lengths).sum()


def update(student, teacher, tokenizer, opt, batches, total_examples, device, force_cap=False):
    import torch
    from sftlib.data import batch
    if total_examples != sum(len(rows) for _, rows in batches):
        raise ValueError("Accumulation denominator mismatch")
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    import time
    started = time.monotonic()
    loss_value, source_tokens, target_tokens = 0.0, 0, 0
    for online, rows in batches:
        inputs, labels = batch(tokenizer, rows, device, limits=PROTOCOL)
        if online:
            labels = generated_labels(generate(student, tokenizer, inputs, True, force_cap), tokenizer)
        valid = labels.ne(-100)
        decoder = {"decoder_input_ids": student.prepare_decoder_input_ids_from_labels(labels=labels),
                   "decoder_attention_mask": torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                teacher_logits = teacher(**inputs, **decoder, use_cache=False).logits
            student_logits = student(**inputs, **decoder, use_cache=False).logits
            loss = forward_kl(student_logits, teacher_logits, labels) / total_examples
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite forward-KL loss")
        loss_value += float(loss.detach())
        loss.backward()
        source_tokens += int(inputs["attention_mask"].sum())
        target_tokens += int(valid.sum())
        del loss, student_logits, teacher_logits, inputs, labels, decoder
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float("inf"), error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    return {"training_seconds": time.monotonic() - started, "loss": loss_value,
            "gradient_norm": float(norm), "source_tokens": source_tokens, "target_tokens": target_tokens,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
