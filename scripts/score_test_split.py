#!/usr/bin/env python
"""Score a finished run's checkpoint on the held-out test split.

Every number logged during training is measured on the *validation* split, which the
staging rule also reads every epoch -- it picks the stage, and under `stop_when_complete`
the epoch at which the final checkpoint is written. This scores the identical benchmark
question (every task at its own full difficulty and its own latent budget) on a third
split that took no part in any of those decisions.

    ./scripts/eval_test.py runs/<name>
    ./scripts/eval_test.py runs/<name> --checkpoint checkpoint_complete.pt --n-test 2000

The run's own `config.json` is the only input besides the checkpoint, so the model, the
vocabulary (`n_nodes` is recorded at train time) and the task family are exactly the ones
that were trained -- nothing is re-specified here and can therefore not drift.

Writes `test_eval_<checkpoint>.json` into the run directory and prints the summary.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from lyot.collator import Collator
from lyot.run import (
    _method_from, _test_tasks_from, accuracy, build_model, score, setup_distributed,
)
from lyot.schedule import build_eval_set
from lyot.stokenizer import STokenizer
from lyot.utils import Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="run directory holding config.json and the checkpoint")
    parser.add_argument("--checkpoint", default="checkpoint_latest.pt",
                        help="which snapshot in run_dir (default: checkpoint_latest.pt)")
    parser.add_argument("--n-test", type=int, default=None,
                        help="test graphs to generate (default: the run's n_valid)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    configs = Config(json.loads((run_dir / "config.json").read_text()))
    if args.n_test is not None:
        configs.set("tasks.n_test", args.n_test)

    dist_ctx = setup_distributed()
    method = _method_from(configs)
    n_nodes = configs.get("n_nodes")
    if not n_nodes:
        raise SystemExit(
            "config.json has no `n_nodes`; it predates that being recorded, so the "
            "vocabulary cannot be reproduced from it. Pass the training config instead.")
    tokenizer = STokenizer(n_nodes=n_nodes, extra_markers=method.extra_markers)
    collate = Collator(tokenizer, latent_id=tokenizer.latent_token_id)

    model = build_model(configs, tokenizer, method)
    blob = torch.load(run_dir / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(blob.get("model", blob))
    model.to(dist_ctx.device).eval()

    tasks = _test_tasks_from(configs)
    # stage=None: the benchmark, not a curriculum diagnostic. `perturb` follows the run so
    # the two splits are scored under the same prompt convention.
    rows = build_eval_set(tasks, method, tokenizer,
                          seed=configs.get("seed", 0),
                          perturb=configs.get("shuffle_nodes", False))
    records = score(model, rows, tokenizer, collate, dist_ctx,
                    max_new_tokens=method.max_new_tokens, batch_size=args.batch_size)
    metrics = accuracy(records)
    # Per-difficulty counts, without which the tail buckets cannot be read: over random
    # trees the depth distribution is concentrated in the middle, so the extremes can hold
    # a handful of graphs and a 0.000 there is one miss, not a finding.
    counts = Counter(int(difficulty) for difficulty, *_ in records)
    n = len(records)
    acc = metrics.get("acc", float("nan"))
    # Normal-approximation half-width at 95%: the split size is a choice, so state what it
    # bought rather than leaving the reader to work it out.
    stderr = math.sqrt(max(acc * (1.0 - acc), 0.0) / n) if n else float("nan")

    trained_epoch = blob.get("epoch")
    print(f"{run_dir.name}  ({method.label}, checkpoint {args.checkpoint}, "
          f"epoch {trained_epoch}, {n} test tasks)")
    print(f"   test/acc {acc:.4f}  +/- {1.96 * stderr:.4f} (95%, normal approx.)")
    for key in sorted((k for k in metrics if k.startswith("acc_d")),
                      key=lambda k: int(k.split("_d")[1])):
        depth = int(key.split("_d")[1])
        print(f"      d{depth:<3} {metrics[key]:.3f}   n={counts.get(depth, 0)}")

    out = run_dir / f"test_eval_{Path(args.checkpoint).stem}.json"
    out.write_text(json.dumps(
        {"run": run_dir.name, "checkpoint": args.checkpoint, "epoch": trained_epoch,
         "method": method.label, "split": tasks.name, "n_tasks": n,
         "seed": configs.get("seed", 0), "acc": acc, "stderr": stderr,
         "n_by_difficulty": {str(k): counts[k] for k in sorted(counts)},
         "metrics": metrics}, indent=2))
    print(f"   wrote {out}")


if __name__ == "__main__":
    main()
