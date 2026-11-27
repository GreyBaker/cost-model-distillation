"""Once-only inference fit/throughput calibration, independent of training fit."""
from __future__ import annotations

import math
from pathlib import Path
import time
import uuid

from .policy import PROTOCOL, VALIDATION_N
from .shared import atomic_json, read_json, sha256, verify_tree
from sftlib.console import message, progress


def request(root, bound, selected, version=None):
    if version is None:
        # Published/unfinished historical calibrations retain their original
        # policy. Updating source code must never force recalibration/rebinding.
        if (root / "evaluation_pilot.json").exists():
            version = read_json(root / "evaluation_pilot.json")["request"]["version"]
        elif (root / "evaluation_pilot_request.json").exists():
            version = read_json(root / "evaluation_pilot_request.json")["version"]
        else:
            version = 2
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported evaluation-pilot policy version")
    result = {"version": version, "binding_sha256": sha256(root / "binding.json"),
            # Current binding() return, NOT binding.json's original code record.
            # Conservative superset of inference sources; provenance, never a gate.
            "source_fingerprints": bound.get("code", {}),
            "training_pilot_sha256": sha256(root / "pilot.json"),
            "runtime": selected["runtime"],
            "source_complete_sha256": bound["inputs"]["hashes"]["student"]["checkpoint"],
            "search_cap": min(bound["settings"]["evaluation_batch_cap"], VALIDATION_N),
            "safety_fraction": bound["settings"].get("evaluation_batch_safety_fraction", .9),
            "reserved_fraction_limit": bound["settings"]["pilot_reserved_fraction_limit"],
            "repeats": bound["settings"]["pilot_updates"],
            "stress": {"source_tokens": 1024, "generation_tokens": 64,
                       "precision": "bfloat16", "sample": False, "force_cap": True,
                        "teacher_loaded": False}}
    if version == 2:
        result.update(selection_policy="warmed-stress-throughput-v1",
                      throughput_near_best_fraction=.98)
    return result


def semantic_request(value):
    """Only source provenance is non-binding; all workload/runtime fields bind."""
    if not isinstance(value, dict):
        raise ValueError("Invalid evaluation-pilot request")
    return {key: item for key, item in value.items() if key != "source_fingerprints"}


def record_provenance(root, expected, requests, action):
    request_source = requests[0].get("source_fingerprints", {})
    sources = [item.get("source_fingerprints", {}) for item in requests[1:]]
    current = expected["source_fingerprints"]
    unique = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    mixed = len(unique) > 1
    changed = any(source != current for source in [request_source, *sources])
    unknown = any(not source for source in [current, request_source, *sources])
    event = {"action": action, "unix_time": time.time(),
             "source_fingerprints": current, "request_source_fingerprints": request_source,
             "calibration_source_fingerprints": unique,
             "mixed_code_calibration": mixed, "differs_from_current_code": changed,
             "unknown_source_provenance": unknown,
             "policy": "Source hashes are provenance only; calibration is reused, not rerun or blocked."}
    atomic_json(root / "evaluation_pilot_provenance" / f"{time.time_ns()}-{uuid.uuid4().hex}.json", event)
    if mixed or changed or unknown:
        message("WARNING: evaluation calibration has mixed, changed or unknown source provenance; "
                "reusing once-only measurements without a source-hash gate. "
                "Fit under current code is not established; inspect evaluation_pilot_provenance/.")


def search_maximum(probe, cap):
    """Largest passing integer below a monotonic fit boundary; bounded by cap."""
    if type(cap) is not int or cap < 1:
        raise ValueError("Positive evaluation search cap required")
    if not probe(1):
        raise RuntimeError("Evaluation batch 1 does not fit with required headroom")
    low = 1
    while low < cap:
        high = min(cap, low * 2)
        if not probe(high):
            break
        low = high
    else:
        return low
    while high - low > 1:
        middle = (high + low) // 2
        if probe(middle):
            low = middle
        else:
            high = middle
    return low


def select_batch(probe, expected):
    maximum = search_maximum(probe, expected["search_cap"])
    batch = max(1, math.floor(maximum * expected["safety_fraction"]))
    # Test the actual integer; preserve each failed safety reduction as evidence.
    while not probe(batch):
        if batch == 1:
            raise RuntimeError("No passing safety-margin evaluation batch")
        batch = max(1, batch // 2)
    return maximum, batch


def speed_candidates(ceiling):
    """Bounded log-spaced sweep plus quarters near the measured safe ceiling."""
    batches = {1, ceiling}
    size = 1
    while size <= ceiling:
        batches.add(size)
        size *= 2
    batches.update(max(1, ceiling * numerator // 4) for numerator in (1, 2, 3))
    return sorted(batches)


def warmed_throughput(result, batch):
    seconds = result["generation_seconds"][1:]  # Exclude the first warmup.
    return batch / (sum(seconds) / len(seconds))


def select_throughput(probe, expected, rate):
    observed = {}
    def measured(batch):
        observed[batch] = probe(batch)
        return observed[batch]
    maximum, ceiling = select_batch(measured, expected)
    for batch in speed_candidates(ceiling):
        measured(batch)
    # Binary fit search can also measure non-sweep batches below the safety
    # ceiling. Do not discard a faster eligible measurement already paid for.
    passing = [batch for batch, passed in observed.items() if passed and batch <= ceiling]
    rates = {batch: rate(batch) for batch in passing}
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError("Invalid warmed evaluation throughput")
    fastest = max(passing, key=lambda batch: (rates[batch], -batch))
    # Avoid spending memory for <=2% timing noise/throughput gain.
    chosen = min(batch for batch in passing
                 if rates[batch] >= rates[fastest] * expected["throughput_near_best_fraction"])
    return maximum, chosen, {
        "safety_ceiling_batch": ceiling,
        "fastest_measured_batch": fastest,
        "fastest_measured_summaries_per_second": rates[fastest],
        "warmed_summaries_per_second": rates[chosen],
    }


def verify_trial(result, expected, batch):
    if (semantic_request(result.get("request", {})) != semantic_request(expected)
            or type(result.get("batch_size")) is not int or result["batch_size"] != batch
            or result.get("status") not in ("pass", "oom", "headroom")):
        raise ValueError("Invalid/foreign evaluation-pilot candidate")
    if result["status"] != "oom":
        if (result.get("runtime") != expected["runtime"]
                or type(result.get("peak_reserved_bytes")) is not int
                or result["peak_reserved_bytes"] < 1):
            raise ValueError("Evaluation-pilot runtime or memory measurement changed")
        limit = expected["runtime"]["total_memory"] * expected["reserved_fraction_limit"]
        if (result["status"] == "pass") != (result["peak_reserved_bytes"] <= limit):
            raise ValueError("Evaluation-pilot headroom status disagrees with measured memory")
        if (len(result.get("generation_seconds", [])) != expected["repeats"]
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                       for x in result["generation_seconds"])):
            raise ValueError("Invalid evaluation-pilot timing coverage")
        if "warmed_summaries_per_second" in result:
            rate = result["warmed_summaries_per_second"]
            if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
                    or not math.isclose(rate, warmed_throughput(result, batch), rel_tol=1e-12)):
                raise ValueError("Reported evaluation throughput differs from measured timings")
    return result["status"] == "pass"


def saved(root, bound, selected):
    """Validate the complete deterministic transcript, without launching probes."""
    try:
        return _saved(root, bound, selected)
    except (AttributeError, KeyError, TypeError, OSError, RuntimeError) as error:
        raise ValueError(f"Invalid/incomplete saved evaluation pilot: {error}") from error


def _saved(root, bound, selected):
    value = read_json(root / "evaluation_pilot.json")
    expected = request(root, bound, selected)
    if semantic_request(value.get("request", {})) != semantic_request(expected):
        raise ValueError("Saved evaluation pilot differs from bound model/settings/runtime")
    trials, results = {}, {}
    for entry in value["result_files"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Saved evaluation-pilot result changed")
        result = read_json(path)
        batch = result["batch_size"]
        if type(batch) is not int or not 1 <= batch <= expected["search_cap"] or batch in trials:
            raise ValueError("Invalid/duplicate evaluation-pilot batch")
        trials[batch] = verify_trial(result, expected, batch)
        results[batch] = result
    visited = set()

    def replay(batch):
        if batch not in trials:
            raise ValueError(f"Saved evaluation pilot is missing required probe batch {batch}")
        visited.add(batch)
        return trials[batch]

    details = {}
    if expected["version"] == 1:
        maximum, batch = select_batch(replay, expected)
    else:
        maximum, batch, details = select_throughput(
            replay, expected, lambda size: warmed_throughput(results[size], size))
    if (type(value["max_passing_batch"]) is not int or value["max_passing_batch"] != maximum
            or type(value["batch_size"]) is not int or value["batch_size"] != batch
            or type(value.get("search_cap_reached")) is not bool
            or value["search_cap_reached"] != (maximum == expected["search_cap"])
            or visited != set(trials)
            or any(value.get(key) != item for key, item in details.items())):
        raise ValueError("Saved evaluation-pilot selection/cap flag differs from deterministic search")
    return value


def ensure(root, bound, selected, runner):
    """Publish one immutable calibration, resume only unfinished candidate search."""
    if (root / "evaluation_pilot.json").exists():
        result = saved(root, bound, selected)
        original_request = (read_json(root / "evaluation_pilot_request.json")
                            if (root / "evaluation_pilot_request.json").exists() else result["request"])
        record_provenance(root, request(root, bound, selected),
                          [original_request, *[read_json(Path(entry["path"]))["request"]
                                               for entry in result["result_files"]]], "reuse")
        message(f"Reusing evaluation pilot: batch {result['batch_size']} "
                f"(max passing {result['max_passing_batch']}).")
        return True
    # Do not change the identity of completed historical evaluations in place.
    legacy = list((root / "baseline").glob("evaluation*/COMPLETE.json"))
    legacy += list((root / "cells").glob("*/evaluation*/COMPLETE.json"))
    if legacy:
        raise ValueError("Completed legacy evaluations have no evaluation pilot; preserve them and use a new grid directory")
    # Recovery identity gains the pilot hash. Refuse BEFORE publishing a request
    # or spending on probes rather than stranding an incomplete legacy run later.
    legacy_batching = list((root / "baseline").glob("evaluation_batching.json"))
    legacy_batching += list((root / "cells").glob("*/evaluation_batching.json"))
    if legacy_batching:
        raise ValueError(f"Legacy evaluation recovery state at {legacy_batching[0]}; "
                         "preserve the old grid and use a new grid directory for calibration")
    expected = request(root, bound, selected)
    request_path = root / "evaluation_pilot_request.json"
    if (request_path.exists()
            and semantic_request(read_json(request_path)) != semantic_request(expected)):
        raise ValueError("Evaluation-pilot request changed; use a new grid directory")
    if not request_path.exists():
        atomic_json(request_path, expected)
    cached_requests = [read_json(request_path)]
    cached_requests += [read_json(path).get("request", {})
                        for path in sorted((root / "evaluation_pilots").glob("batch-*.json"))]
    record_provenance(root, expected, cached_requests, "search_start_or_resume")
    # Exponential + binary search plus halving safety chain. The powers in the
    # throughput sweep are already capacity probes; at most three quarters add work.
    probe_bound = 3 * math.ceil(math.log2(expected["search_cap"] + 1)) + 2
    if expected["version"] == 2:
        probe_bound += 3
    message(f"Evaluation calibration: at most {probe_bound} distinct candidate subprocesses "
            f"({probe_bound * expected['repeats']} full 1024/forced-64 stress repetitions). "
            "Cached candidates cost no generation; each fresh process loads the student, "
            "verifies checkpoint hashes and scans validation lengths.")
    files, results = {}, {}

    def probe(batch):
        output = root / "evaluation_pilots" / f"batch-{batch:05d}.json"
        if not output.exists():
            if not runner.call("evaluation_pilot", "shared", "--microbatch", batch, "--output", output):
                raise InterruptedError("Evaluation pilot paused; completed candidates are preserved")
        result = read_json(output)
        passed = verify_trial(result, expected, batch)
        results[batch] = result
        files[batch] = {"path": str(output), "sha256": sha256(output)}
        return passed

    try:
        details = {}
        if expected["version"] == 1:
            maximum, batch = select_batch(probe, expected)
        else:
            maximum, batch, details = select_throughput(
                probe, expected, lambda size: warmed_throughput(results[size], size))
    except InterruptedError:
        return False
    atomic_json(root / "evaluation_pilot.json", {
        "request": expected, "max_passing_batch": maximum, "batch_size": batch,
        "search_cap_reached": maximum == expected["search_cap"],
        **details,
        "selection": ("largest measured stress-fit batch with reserved headroom, then batch safety fraction"
                      if expected["version"] == 1 else
                      "smallest measured batch within 2% of best warmed throughput, below tested safety ceiling"),
        "limitation": "Bounded stress-workload sweep, not an exhaustive real-pool throughput optimum; "
                      "later checkpoint OOMs retain bounded fallback",
        "result_files": [files[k] for k in sorted(files)]})
    saved(root, bound, selected)
    record_provenance(root, expected,
                      [read_json(request_path), *[read_json(Path(entry["path"]))["request"]
                                                 for entry in files.values()]], "published")
    message(f"Evaluation pilot selected shared batch {batch}; max passing {maximum}, "
            f"safety fraction {expected['safety_fraction']:.0%}."
            + (f" Warmed stress throughput {details['warmed_summaries_per_second']:.2f} summaries/s."
               if details else ""))
    return True


def worker(root, bound, selected, batch, output):
    """One candidate in a fresh CUDA process; no teacher, optimizer or backward."""
    started = time.monotonic()
    import torch
    from sftlib import data, modeling
    from . import backend
    expected = request(root, bound, selected)
    if type(batch) is not int or not 1 <= batch <= expected["search_cap"]:
        raise ValueError("Invalid evaluation-pilot batch")
    if output.exists():
        raise FileExistsError("Do not overwrite completed evaluation-pilot candidates")
    try:
        directory = Path(bound["inputs"]["paths"]["student_checkpoint"])
        marker = read_json(directory / "COMPLETE.json")
        if sha256(directory / "COMPLETE.json") != expected["source_complete_sha256"]:
            raise ValueError("Evaluation-pilot source checkpoint changed")
        verify_tree(directory, marker["files"])
        device, runtime = backend.runtime(bound["settings"])
        if runtime != expected["runtime"]:
            raise ValueError("Evaluation-pilot GPU/software differs from training pilot")
        tokenizer, model = modeling.load(directory, device, False,
                                         {**PROTOCOL, "gradient_checkpointing": False})
        dataset = data.load(Path(bound["inputs"]["sft_work"]) / "student")["validation"]
        if len(dataset) != VALIDATION_N or len(set(map(str, dataset["id"]))) != VALIDATION_N:
            raise ValueError("Wrong validation pool for evaluation pilot")
        longest = max(range(len(dataset)), key=lambda i: len(dataset[i]["input_ids"]))
        tokens = dataset[longest]["input_ids"]
        if not tokens or len(tokens) > 1024:
            raise ValueError("Invalid prepared source length")
        content = tokens[:-1] or [tokenizer.eos_token_id]
        # Always stress the full protocol shape even if the actual pool is shorter.
        stress = {**dataset[longest],
                  "input_ids": (content * math.ceil(1023 / len(content)))[:1023] + [tokenizer.eos_token_id],
                  "labels": [tokenizer.eos_token_id] * 64}
        torch.cuda.reset_peak_memory_stats(device)
        inputs, labels = data.batch(tokenizer, [stress] * batch, device)
        seconds = []
        with progress(expected["repeats"], desc=f"Eval pilot batch={batch}", unit="generation") as bar:
            for _ in range(expected["repeats"]):
                torch.cuda.synchronize(device)
                before = time.monotonic()
                ids = backend.generate(model, tokenizer, inputs, sample=False, force_cap=True)
                if tuple(ids.shape) != (batch, 64):
                    raise ValueError("Evaluation pilot did not exercise the full generation cap")
                # Match evaluation's GPU→CPU output transfer as well as generation.
                ids.cpu().tolist()
                del ids
                torch.cuda.synchronize(device)
                seconds.append(time.monotonic() - before)
                bar.update()
        peak = torch.cuda.max_memory_reserved(device)
        result = {"status": "pass" if peak <= runtime["total_memory"] * expected["reserved_fraction_limit"]
                  else "headroom", "runtime": runtime, "peak_reserved_bytes": peak,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "generation_seconds": seconds,
                  "warmed_summaries_per_second": batch / (sum(seconds[1:]) / (len(seconds) - 1)),
                  "stress_source_id": str(dataset[longest]["id"])}
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "error": str(error)}
    atomic_json(output, {**result, "request": expected, "batch_size": batch,
                         "wall_seconds": time.monotonic() - started})
