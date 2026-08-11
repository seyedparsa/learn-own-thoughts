#!/usr/bin/env python
"""Backward cost against the truncation window: seconds per step and peak memory.

The training logs cannot answer this cleanly. `time_s` is per *epoch* and includes
evaluation -- which scores every stage and is identical across arms -- so the fixed cost
dilutes the ratio and understates the saving; arms also shared GPUs, so absolute seconds
are contended. Peak memory was never recorded at all.

This measures one training step directly, at a fixed stage and batch, sweeping the window:

    ./scripts/bench_backward.py configs/tree20/atc.yaml
    ./scripts/bench_backward.py configs/tree20/atc.yaml --stage 16 --windows full,8,5,4,2,1

Every window is measured in the same process on the same batch, so the comparison is
internally consistent even if the machine is busy. `torch.cuda.max_memory_allocated` is
per-process, so the memory column is unaffected by other jobs; the timing column is not,
and the script says so when it cannot get an idle device.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lyot.collator import Collator
from lyot.run import _method_from, _tasks_from, build_model, forward_loss
from lyot.schedule import build_training_set
from lyot.stokenizer import STokenizer
from lyot.utils import Config


def measure(model, batch, optimizer, steps: int) -> tuple[float, int]:
    """Median seconds per full training step, and peak bytes allocated."""
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps + 2):          # first two are warmup (autograd graph, allocator)
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = forward_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - started)
    times.sort()
    return times[len(times) // 2], torch.cuda.max_memory_allocated()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_file")
    parser.add_argument("--stage", type=int, default=None,
                        help="curriculum stage to build the batch at (default: max_latent_stage)")
    parser.add_argument("--windows", default="full,8,5,4,2,1")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=None, help="write the table as JSON here")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: peak-memory accounting and step timing are both CUDA")

    configs = Config.from_yaml(args.config_file, [])
    stage = args.stage if args.stage is not None else configs.get("max_latent_stage", 4)
    batch_size = args.batch_size or configs.get("batch_size_training", 128)
    # A small split is enough -- one batch is measured, and the graphs' depth distribution is
    # what sets the latent counts, not how many of them there are.
    configs.set("tasks.n_train", max(batch_size * 4, 512))
    configs.set("tasks.n_valid", 8)

    train_tasks, val_tasks = _tasks_from(configs)
    n_nodes = configs.get("n_nodes") or 1 + max(
        node for tasks in (train_tasks, val_tasks) for t in tasks for e in t.edges for node in e)
    configs.set("n_nodes", n_nodes)
    method = _method_from(configs)
    tokenizer = STokenizer(n_nodes=n_nodes, extra_markers=method.extra_markers)
    collate = Collator(tokenizer, latent_id=tokenizer.latent_token_id)
    rows = build_training_set(train_tasks, method, tokenizer, stage=stage,
                             uniform_prob=configs.get("uniform_prob", 0.0), seed=0)
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v)
             for k, v in collate(rows[:batch_size]).items()}
    latents = int((batch["input_ids"] == tokenizer.latent_token_id).sum(1).max())

    model = build_model(configs, tokenizer, method).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)   # tiny: we measure cost, not learning

    windows = [None if w.strip() == "full" else int(w) for w in args.windows.split(",")]
    print(f"{Path(args.config_file).name}  stage {stage}, batch {batch_size}, "
          f"{latents} latent passes in the widest row, {args.steps} timed steps")
    print(f"{'window b':>10} {'s / step':>10} {'peak mem':>12} {'vs full: time':>14} {'mem':>8}")
    results, base = {}, None
    for w in windows:
        model.backprop_last = w
        seconds, peak = measure(model, batch, optimizer, args.steps)
        if base is None:
            base = (seconds, peak)
        label = "full" if w is None else str(w)
        results[label] = {"seconds_per_step": seconds, "peak_bytes": peak}
        print(f"{label:>10} {seconds*1e3:>9.1f}ms {peak/2**30:>10.2f}GiB "
              f"{seconds/base[0]:>13.2f}x {peak/base[1]:>7.2f}x")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"config": args.config_file, "stage": stage, "batch_size": batch_size,
             "latent_passes": latents, "steps": args.steps, "results": results}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
