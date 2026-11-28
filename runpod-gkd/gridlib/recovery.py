"""Bounded CUDA-OOM recovery; no message matching, recipe changes or in-process retry."""
from __future__ import annotations

import time

from .shared import atomic_json, read_json, sha256

OOM_EXIT_CODE = 86


class CudaOOM(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"CUDA OOM in {report['operation']} for {report['cell']}")


def identity(root, evaluation=False):
    value = {"binding_sha256": sha256(root / "binding.json"),
             "pilot_sha256": sha256(root / "pilot.json")}
    if evaluation:
        value["evaluation_pilot_sha256"] = sha256(root / "evaluation_pilot.json")
    return value


def read_batch(root, path, initial, training=False):
    if not path.exists():
        return initial, []
    saved = read_json(path)
    value = saved["microbatch"]
    if (saved["identity"] != identity(root, evaluation=not training) or type(value) is not int or not 1 <= value <= initial
            or (training and 32 % value)):
        raise ValueError("Foreign/invalid saved batching recovery state")
    return value, saved["history"]


def training_microbatch(root, selected):
    return read_batch(root, root / "training_batching.json", selected["microbatch"], True)[0]


def evaluation_path(root, item):
    from .schedule import evaluation_directory
    owner = evaluation_directory(root, item)
    return owner / "evaluation_batching.json"


def evaluation_initial(root, bound, selected):
    from .evalpilot import saved
    return saved(root, bound, selected)["batch_size"]


def evaluation_microbatch(root, bound, item, selected):
    initial = evaluation_initial(root, bound, selected)
    return read_batch(root, evaluation_path(root, item), initial)[0]


def lower(root, path, initial, failed, *, training=False, **context):
    current, history = read_batch(root, path, initial, training)
    if current != failed:
        raise ValueError("Failed attempt does not match active microbatch")
    if current == 1:
        raise RuntimeError("CUDA OOM at microbatch 1; no further automatic reduction is possible")
    smaller = max(1, current // 2)
    event = {"from_microbatch": current, "to_microbatch": smaller, "unix_time": time.time(), **context}
    atomic_json(path, {"identity": identity(root, evaluation=not training),
                       "microbatch": smaller, "history": [*history, event]})
    print(f"CUDA OOM: retry in a fresh process at microbatch {smaller} (was {current}).", flush=True)
    return smaller


def checkpoint_for_retry(root, item):
    from .milestones import latest_restart
    return latest_restart(root, read_json(root / "binding.json"), item,
                          read_json(root / "pilot.json")["selected"])


def train_with_recovery(root, item, selected, runner, until_points=None):
    while True:
        microbatch = training_microbatch(root, selected)
        try:
            extra = [] if until_points is None else ["--until-points", until_points]
            return runner.call("train", item["id"], "--cell", item["id"], "--microbatch", microbatch, *extra)
        except CudaOOM as error:
            # No incomplete gradients/optimizer updates are salvaged from the failed worker.
            # The new worker restores ONLY this cell's most recent complete checkpoint.
            checkpoint, marker = checkpoint_for_retry(root, item)
            lower(root, root / "training_batching.json", selected["microbatch"], microbatch,
                  training=True, cell=item["id"], checkpoint=str(checkpoint), step=marker["step"],
                  failure=error.report, effective_batch_size=32)
