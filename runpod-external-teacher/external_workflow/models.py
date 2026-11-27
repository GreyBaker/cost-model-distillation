"""Real GPU probability/gradient probes and native model export."""
from __future__ import annotations

from pathlib import Path

from .acquire import source, student_source, verified_model_source
from .common import (PROFILE, atomic_json, cached, check_space, identity, pair_root, publish, read_json,
                     request, runtime, sha256, temporary, verify)
from .data import load as load_data, tokenizer as load_tokenizer
from gridlib.policy import PROTOCOL


def inventory(model):
    groups, storage, total = {}, {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append(name)
        key = (parameter.untyped_storage().data_ptr(), parameter.storage_offset(),
               tuple(parameter.shape), str(parameter.dtype))
        storage.setdefault(key, []).append(name)
        total[id(parameter)] = parameter.numel()
    return {"parameter_aliases": sorted(sorted(x) for x in groups.values() if len(x) > 1),
            "storage_aliases": sorted(sorted(x) for x in storage.values() if len(x) > 1),
             "unique_parameters": sum(total.values()),
             "shapes": {name: list(p.shape) for name, p in model.named_parameters()},
             "dtypes": {name: str(p.dtype) for name, p in model.named_parameters()},
             "buffers": {name: {"shape": list(p.shape), "dtype": str(p.dtype)}
                         for name, p in model.named_buffers()},
            "scale_decoder_outputs": bool(getattr(model.config, "scale_decoder_outputs",
                                                    model.config.tie_word_embeddings))}


def normalize_export_config(model):
    """Describe actual untied parameters without changing weights or decoder scaling."""
    before = inventory(model)
    if before["scale_decoder_outputs"] is not False:
        raise ValueError("Config normalization must not change decoder logit scaling")
    source_config = {
        "tie_word_embeddings": model.config.tie_word_embeddings,
        "scale_decoder_outputs": getattr(model.config, "scale_decoder_outputs", None),
    }
    embeddings, output = model.get_input_embeddings(), model.get_output_embeddings()
    if embeddings is None or output is None:
        raise ValueError("T5 export requires input and output embeddings")
    tied = embeddings.weight is output.weight
    if not tied:
        if embeddings.weight.untyped_storage().data_ptr() == output.weight.untyped_storage().data_ptr():
            raise ValueError("Distinct input/output parameters share storage; cannot normalize as untied")
        model.config.tie_word_embeddings = False
    # Older restored configs may encode this through tie_word_embeddings=False.
    # Make the already-verified behavior explicit for the native derived export.
    model.config.scale_decoder_outputs = False
    if inventory(model) != before:
        raise ValueError("Config normalization changed parameter aliases, dimensions or scaling")
    return {
        "kind": "actual-input-output-parameter-config-v1",
        "source_loaded_config": source_config,
        "export_config": {"tie_word_embeddings": model.config.tie_word_embeddings,
                          "scale_decoder_outputs": model.config.scale_decoder_outputs},
        "input_output_parameter_tied": tied,
        "note": "Derived config only; source files and parameter/storage aliases are unchanged.",
    }


def load_model(path, device, precision="float32", reference=False):
    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    cfg.dropout_rate = 0.0
    if cfg.model_type != "t5" or not cfg.is_encoder_decoder:
        raise ValueError("Only T5 encoder-decoder sources are supported")
    dtype = torch.float32 if precision == "float32" else torch.bfloat16
    options = {"torch_dtype" if reference else "dtype": dtype}
    model, info = AutoModelForSeq2SeqLM.from_pretrained(
        path, config=cfg, local_files_only=True, trust_remote_code=False, weights_only=True,
        device_map={"": str(device)}, attn_implementation="eager", output_loading_info=True, **options)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Source load did not preserve all tensors: {info}")
    if precision == "bfloat16":
        model.to(torch.bfloat16)
        model.requires_grad_(False)
    model.eval()
    if inventory(model)["scale_decoder_outputs"]:
        raise ValueError("LM-adapted external workflow requires unscaled decoder logits; do not edit this away")
    if (model.config.vocab_size != 32128 or model.config.decoder_start_token_id != 0
            or model.config.pad_token_id != 0 or model.config.eos_token_id != 1):
        raise ValueError("Unexpected T5 vocabulary/start token")
    return model


def probe_rows(work):
    table = load_data(work)["validation"]
    # Several real source lengths; all decoder positions on reference and
    # source-student-generated trajectories are checked.
    indices = [0, 1]
    indices += [max(range(len(table)), key=lambda i: len(table[i]["input_ids"]))]
    indices += [len(table) // 2]
    return [table[index] for index in dict.fromkeys(indices)]


def make_inputs(model, tokenizer, rows, device):
    from sftlib.data import batch
    inputs, labels = batch(tokenizer, rows, device, limits=PROTOCOL)
    return inputs, labels


def outputs(model, inputs, labels):
    import torch
    valid = labels.ne(-100)
    decoder = {"decoder_input_ids": model.prepare_decoder_input_ids_from_labels(labels=labels),
               "decoder_attention_mask": torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], dim=1)}
    return model(**inputs, **decoder, use_cache=False).logits[valid].float()


def greedy(model, tokenizer, inputs, bf16=False):
    """Same explicit generation protocol, but genuinely FP32 for FP32 probes."""
    import torch
    from contextlib import nullcontext
    from transformers import GenerationConfig
    config = GenerationConfig(max_new_tokens=64, do_sample=False, num_beams=1,
        min_new_tokens=0, no_repeat_ngram_size=0, repetition_penalty=1.0,
        length_penalty=1.0, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id, use_cache=True)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    with torch.no_grad(), context:
        ids = model.generate(**inputs, generation_config=config)
    if not ids[:, 0].eq(model.config.decoder_start_token_id).all():
        raise ValueError("Unexpected probe decoder start")
    return ids[:, 1:]


def tensors_and_gradients(model, tokenizer, rows, device, gradient=True, fixed=None, bf16=False):
    import torch
    import torch.nn.functional as F
    from contextlib import nullcontext
    from sftlib import modeling
    from gridlib.backend import generated_labels
    inputs, labels = make_inputs(model, tokenizer, rows, device)
    model.eval()
    model.zero_grad(set_to_none=True)
    generated = (fixed["generated"].to(device) if fixed is not None else
                  greedy(model, tokenizer, inputs, bf16=bf16))
    on_labels = generated_labels(generated, tokenizer)
    result = {**{key: value.cpu().contiguous() for key, value in inputs.items()},
              "labels": labels.cpu().contiguous(), "on_labels": on_labels.cpu().contiguous(),
              "generated": generated.cpu().contiguous()}
    context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
    with context:
        for name, target in (("reference_logits", labels), ("on_policy_logits", on_labels)):
            with torch.no_grad():
                logits = outputs(model, inputs, target)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite probability probe")
                result[name] = logits.cpu().contiguous()
                result[name.replace("_logits", "_ce")] = F.cross_entropy(
                    logits, target[target.ne(-100)], reduction="mean").cpu().contiguous()
        if gradient:
            loss = modeling.loss_sum(model, inputs, labels) / int(labels.ne(-100).sum())
            loss.backward()
    if gradient:
        for name, param in model.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"Absent/nonfinite gradient: {name}")
            result["gradient." + name] = param.grad.float().cpu().contiguous()
    model.zero_grad(set_to_none=True)
    return result


def compare(actual, expected, rtol=1e-4, atol=1e-4, gradients=True, generated=True):
    import torch
    if set(actual) != set(expected):
        raise ValueError("Probe tensor/gradient keys changed")
    errors, probabilities = {}, {}
    for key in actual:
        if not gradients and key.startswith("gradient."):
            continue
        if key == "generated" and not generated:
            continue
        a, b = actual[key], expected[key]
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            errors[key] = float((a - b).abs().max())
            if key.endswith("_logits"):
                p, q = b.log_softmax(-1), a.log_softmax(-1)
                torch.testing.assert_close(q, p, rtol=rtol, atol=atol)
                kl = (p.exp() * (p - q)).sum(-1)
                if not torch.isfinite(kl).all():
                    raise FloatingPointError("Nonfinite probability comparison")
                probabilities[key] = {
                    "rms_logits_error": float((a - b).square().mean().sqrt()),
                    "max_log_probability_error": float((p - q).abs().max()),
                    "mean_forward_kl": float(kl.mean()), "max_forward_kl": float(kl.max()),
                }
        elif not torch.equal(a, b):
            raise ValueError(f"Input, mask, label or greedy trajectory changed: {key}")
    return {"max_abs_error": max(errors.values(), default=0), "rtol": rtol, "atol": atol,
            "probabilities": probabilities}


def legacy_probe(work):
    from safetensors.torch import save_file
    final = pair_root(work) / "reports/legacy_probe"
    expected = identity(work, "legacy_probe")
    if cached(final, expected):
        return
    device, info = runtime(work, reference=True, gpu=True)
    model = load_model(verified_model_source(work, "student"), device, reference=True)
    tok = load_tokenizer(student_source(work))
    tensors = tensors_and_gradients(model, tok, probe_rows(work), device)
    scratch = temporary(final)
    save_file(tensors, scratch / "probe.safetensors")
    atomic_json(scratch / "report.json", {"inventory": inventory(model), "runtime": info,
                "source_complete_sha256": sha256(student_source(work) / "COMPLETE.json")})
    publish(scratch, final, expected)


def export(work, role):
    import torch
    from safetensors.torch import load_file, save_file
    directory = pair_root(work)
    final = directory / "models" / f"{role}_{'fp32' if role == 'student' else 'bf16'}"
    expected = identity(work, "model", role=role)
    if cached(final, expected):
        return
    device, info = runtime(work, gpu=True)
    source_path = verified_model_source(work, role)
    canonical_tokenizer = load_tokenizer(student_source(work))
    model = load_model(source_path, device)
    before_inventory = inventory(model)
    rows = probe_rows(work)
    reference = None
    if role == "student" and request(work)["definition"]["student"] == "restored":
        legacy = directory / "reports/legacy_probe"
        verify(legacy, identity(work, "legacy_probe"))
        legacy_report = read_json(legacy / "report.json")
        if before_inventory != legacy_report["inventory"]:
            raise ValueError("Restored student parameterization/aliases/scaling changed in native runtime")
        reference = load_file(legacy / "probe.safetensors")
    before = tensors_and_gradients(model, canonical_tokenizer, rows, device, gradient=role == "student",
                                    fixed=reference)
    comparison = compare(before, reference) if reference is not None else {"native_source": True}
    # Separately generate rather than merely reusing the legacy trajectories.
    if reference is not None:
        inputs, _ = make_inputs(model, canonical_tokenizer, rows, device)
        if not torch.equal(greedy(model, canonical_tokenizer, inputs).cpu(), reference["generated"]):
            raise ValueError("Cross-version greedy student trajectories changed")
    model.config.external_pair_profile = PROFILE
    model.config.use_cache = True
    normalization = normalize_export_config(model)
    normalized = tensors_and_gradients(model, canonical_tokenizer, rows, device,
                                       gradient=role == "student", fixed=before)
    normalization["parity"] = compare(normalized, before)
    del normalized
    inputs, _ = make_inputs(model, canonical_tokenizer, rows, device)
    if not torch.equal(greedy(model, canonical_tokenizer, inputs).cpu(), before["generated"]):
        raise ValueError("Config normalization changed greedy generation")
    if role == "teacher":
        model.requires_grad_(False)
        model.to(torch.bfloat16)
        rounded = tensors_and_gradients(model, canonical_tokenizer, rows, device, gradient=False, bf16=True)
        drift = {}
        fixed = tensors_and_gradients(model, canonical_tokenizer, rows, device,
                                       gradient=False, fixed=before, bf16=True)
        for key in ("reference_logits", "on_policy_logits"):
            # Different generated prefixes have different lengths: compare
            # rounding on fixed source-model trajectories as well.
            p = before[key].log_softmax(-1)
            q = fixed[key].log_softmax(-1)
            kl = (p.exp() * (p - q)).sum(-1)
            if not torch.isfinite(kl).all():
                raise FloatingPointError("Nonfinite FP32/BF16 probability drift")
            drift[key] = {"mean_forward_kl": float(kl.mean()), "max_forward_kl": float(kl.max()),
                          "max_abs_logits": float((before[key] - fixed[key]).abs().max()),
                          "fp32_entropy": float(-(p.exp() * p).sum(-1).mean()),
                          "bf16_entropy": float(-(q.exp() * q).sum(-1).mean())}
        source_text = canonical_tokenizer.batch_decode(before["generated"], skip_special_tokens=True)
        rounded_text = canonical_tokenizer.batch_decode(rounded["generated"], skip_special_tokens=True)
        drift["greedy_summaries"] = {"fp32": source_text, "bf16": rounded_text,
                                    "changed": sum(a != b for a, b in zip(source_text, rounded_text))}
        target_probe = rounded
    else:
        target_probe = before
        drift = None
    check_space(final.parent, int(sum(p.numel() * p.element_size() for p in model.parameters()) * 1.2) + 1024 ** 3)
    scratch = temporary(final)
    model.save_pretrained(scratch, safe_serialization=True, max_shard_size="2GB")
    canonical_tokenizer.save_pretrained(scratch)
    saved_inventory = inventory(model)
    del model
    torch.cuda.empty_cache()
    reloaded = load_model(scratch, device, precision="float32" if role == "student" else "bfloat16")
    if any(getattr(reloaded.config, key, None) != value
           for key, value in normalization["export_config"].items()):
        raise ValueError("Native serialization changed the normalized embedding/scaling config")
    after_inventory = inventory(reloaded)
    if saved_inventory != after_inventory:
        raise ValueError("Native serialization changed parameter aliases, dimensions or scaling")
    after = tensors_and_gradients(reloaded, canonical_tokenizer, rows, device, gradient=role == "student",
                                  fixed=target_probe, bf16=role == "teacher")
    parity = compare(after, target_probe)
    inputs, _ = make_inputs(reloaded, canonical_tokenizer, rows, device)
    if not torch.equal(greedy(reloaded, canonical_tokenizer, inputs, bf16=role == "teacher").cpu(),
                       target_probe["generated"]):
        raise ValueError("Same-precision export/reload generation changed")
    save_file({k: v for k, v in after.items() if not k.startswith("gradient.")}, scratch / "probe.safetensors")
    atomic_json(scratch / "provenance.json", {
        "role": role, "source": str(source_path), "source_complete_sha256": sha256(source_path / "COMPLETE.json"),
        "origin": "restored-local-sft" if role == "student" and reference is not None else "external-public-sft",
        "local_sft_updates": 0, "historical_steps": 100000 if role == "student" else 8000,
        "historical_steps_status": "verified-local-completion" if reference is not None else "publisher-reported",
        "profile": PROFILE, "runtime": info, "inventory": after_inventory,
        "config_normalization": normalization,
        "probe_generation_precision": "bfloat16" if role == "teacher" else "float32",
        "migration": comparison, "reload_parity": parity, "fp32_bf16_drift": drift,
        "tokenizer": "audited canonical student tokenizer, exact token-ID identity"})
    publish(scratch, final, expected, role=role, precision="float32" if role == "student" else "bfloat16")
