"""Training loop for continuous-thought curricula.

Modern rewrite of the upstream ``run.py`` (kept verbatim alongside this file).
Runs unchanged on one GPU, several GPUs, or CPU::

    python -m lyot.run configs/prosqa_coconut.yaml            # single device
    torchrun --nproc_per_node 4 -m lyot.run configs/...yaml   # data parallel
    python -m lyot.run configs/...yaml lr=3e-4 num_epochs=5   # with overrides

Differences from upstream, beyond portability:

* Upstream calls ``dist.init_process_group`` unconditionally and reads ``LOCAL_RANK``
  from the environment, so plain ``python run.py`` dies immediately. Here the launch
  mode is detected, and every collective is a no-op when running on one device.
* Datasets are rebuilt only when the curriculum stage changes, not every epoch --
  upstream re-tokenised all ~15k samples per epoch even though the stage moves once
  every ``epochs_per_stage``.
* DDP rather than FSDP. Upstream builds an FSDP policy that shards only
  ``LlamaDecoderLayer``, so for a GPT-2 backbone nothing is sharded and it already
  behaves as DDP; at 15M parameters sharding costs more than it saves.
* Evaluation reports accuracy per difficulty, which is what a curriculum claim needs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from lyot.coconut import Coconut
from lyot.collator import Collator
from lyot.schedule import (
    build_eval_set, build_training_set, next_stage, scheduled_stage,
)
from lyot.methods import Method, build as build_method
from lyot.stokenizer import STokenizer
from lyot.tasks import TaskSet, generate, load_tasks
from lyot.utils import Config, set_seed


# -- distributed plumbing ----------------------------------------------------------

@dataclass(frozen=True)
class Dist:
    """Launch topology. Single-device runs get rank 0 of world size 1, so the rest of
    the file never branches on whether torchrun was used."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    def sum_reduce(self, value: float | int) -> float:
        """Sum a scalar across ranks (identity when single-device)."""
        if not self.is_distributed:
            return float(value)
        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def gather_objects(self, obj: Any) -> list[Any]:
        if not self.is_distributed:
            return [obj]
        bucket: list[Any] = [None] * self.world_size
        dist.all_gather_object(bucket, obj)
        return bucket


def setup_distributed() -> Dist:
    """torchrun sets LOCAL_RANK; its absence means a plain single-process run."""
    if "LOCAL_RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return Dist(device=device)

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return Dist(
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
        device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"),
    )


def _tasks_from(configs: Config) -> tuple[TaskSet, TaskSet]:
    """The task half of a run: train and val splits.

    Either loaded from files, or generated on the fly when the config names a graph
    family -- ``tasks: {family: random_tree, n: 100, n_train: 4000, n_valid: 200}``.
    Generating avoids shipping a corpus per family, and keeps the two splits' graphs
    disjoint by seeding them apart.
    """
    spec = configs.get("tasks")
    if spec:
        spec = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        family = spec.pop("family")
        n_train = spec.pop("n_train", 2000)
        n_valid = spec.pop("n_valid", 200)
        seed = spec.pop("seed", configs.get("seed", 0))
        return (_generate_mixed(family, n_train, seed, f"{family}-train", spec),
                # A different seed, so no validation graph was trained on.
                _generate_mixed(family, n_valid, seed + 10_000, f"{family}-valid", spec))

    limit = configs.get("limit_samples")
    loader = configs.get("task_family", "reachability")
    return (load_tasks(configs.train_path, family=loader, limit=limit),
            load_tasks(configs.val_path, family=loader, limit=limit))


def _test_tasks_from(configs: Config) -> TaskSet:
    """The third split -- graphs no part of the run has looked at.

    Everything logged during training is measured on the *validation* split, and the
    staging rule reads that same split every epoch: it decides the stage, and under
    `stop_when_complete` the epoch the final checkpoint is written at. So `eval/acc` is
    model-selection data by construction, and a headline number should not come from it.

    Same family and options, a third RNG stream: `n_test` graphs (defaulting to `n_valid`)
    at `seed + 20_000`. Deterministic in `seed`, so the split is reconstructible from a
    run's own config.json and no corpus has to be shipped beside the checkpoint.

    The offsets are spaced by 10_000 while `_generate_mixed` spaces a list-valued option's
    members by 1_000, so the three streams stay disjoint for up to ten list members.
    """
    spec = configs.get("tasks")
    if not spec:
        path = configs.get("test_path")
        if not path:
            raise ValueError(
                "no test split available: a file-based run needs `test_path`, a generated "
                "one needs a `tasks:` block")
        return load_tasks(path, family=configs.get("task_family", "reachability"),
                          limit=configs.get("limit_samples"))
    spec = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    family = spec.pop("family")
    spec.pop("n_train", None)
    n_valid = spec.pop("n_valid", 200)
    n_test = spec.pop("n_test", n_valid)
    seed = spec.pop("seed", configs.get("seed", 0))
    return _generate_mixed(family, n_test, seed + 20_000, f"{family}-test", spec)


def _generate_mixed(family: str, n_graphs: int, seed: int, name: str,
                    spec: dict) -> TaskSet:
    """Generate a split, allowing any family option to be a list.

    ``length: [1, 2, 3, 4]`` yields a mixture of graph sizes in one split, which is what
    makes a breadth curriculum expressible: `max_edges` then admits the small graphs first
    and the large ones later. With a single fixed size there is nothing to grow.
    """
    from lyot.tasks.base import TaskSet

    varied = {k: v for k, v in spec.items() if isinstance(v, (list, tuple))}
    if not varied:
        return generate(family, n_graphs=n_graphs, seed=seed, name=name, **spec)
    if len(varied) > 1:
        raise ValueError(f"only one option may be a list, got {sorted(varied)}")
    key, values = next(iter(varied.items()))
    fixed = {k: v for k, v in spec.items() if k != key}
    per = max(1, n_graphs // len(values))
    tasks = []
    for i, value in enumerate(values):
        part = generate(family, n_graphs=per, seed=seed + 1000 * i, **{key: value}, **fixed)
        tasks.extend(part.tasks)
    return TaskSet(name=name, tasks=tasks)


def _edge_budget(stage: int, configs: Config) -> Optional[int]:
    """Edge ceiling for this stage, when growing search breadth over training.

    `grow_edges: {start: 8, per_stage: 4}` admits 8-edge graphs at stage 0 and four more
    each stage. Breadth is what actually binds answer-only supervision -- a one-hop lookup
    is learned at 0.997 with 8 edges and 0.53 with 12 -- and no other axis we vary moves it.
    """
    spec = _as_dict(configs.get("grow_edges", {}))
    if not spec:
        return None
    return spec.get("start", 8) + stage * spec.get("per_stage", 4)


def _as_dict(value) -> dict:
    """Config wraps nested mappings, so a `method_options:` block arrives as a Config and
    cannot be **-expanded."""
    return value.to_dict() if hasattr(value, "to_dict") else dict(value or {})


def lr_schedule_value(epoch: int, *, peak: float, total_epochs: int,
                      kind: str = "constant", warmup: int = 0,
                      floor: float = 1.0e-6) -> float:
    """Learning rate at ``epoch``, as a pure function so the schedule is testable.

    ``warmup`` epochs ramp linearly from ``peak/warmup`` up to ``peak`` (standard linear
    warmup, orthogonal to the schedule). After warmup, ``kind`` shapes the rest: ``constant``
    holds ``peak``; ``cosine`` decays ``peak`` -> ``floor`` over the remaining epochs. The lr
    is set per epoch, and at ~78 steps/epoch a few warmup epochs is a few hundred steps.
    """
    if warmup and epoch < warmup:
        return peak * (epoch + 1) / warmup
    if kind == "cosine":
        span = max(total_epochs - 1 - warmup, 1)
        import math as _m
        cos = 0.5 * (1.0 + _m.cos(_m.pi * min(max(epoch - warmup, 0), span) / span))
        return floor + (peak - floor) * cos
    if kind != "constant":
        raise ValueError(f"unknown lr_schedule {kind!r}; have constant, cosine")
    return peak


def should_reset_optimizer(mode, epoch: int, stage_changed: bool) -> bool:
    """Whether to rebuild the optimizer this epoch, as a pure function so it is testable.

    ``reset_optimizer`` takes three forms, all handled here:
      * ``false`` (default) -- never reset; Adam's moment estimates carry across the whole run.
      * ``true`` -- reset every epoch, clearing the moments even within a stage.
      * ``"stage"`` -- reset only when the stage changes. The moment estimates are valid while
        the objective is fixed, so this keeps them within a stage and clears the stale first/
        second moments only when the stage (hence the loss landscape) moves. This is the
        middle ground between the two: less disruptive than every-epoch, but still drops the
        momentum built for the old stage's gradients when a new stage's arrive.

    Epoch 0 never resets under any mode -- ``fresh_optimizer()`` already built it, and a
    staging run reports ``stage_changed`` on its first real transition, not at start.
    """
    if not epoch:
        return False
    if mode is True:
        return True
    if isinstance(mode, str) and mode.strip().lower() in ("stage", "on_stage_change"):
        return stage_changed
    return False


def _method_from(configs: Config) -> Method:
    """The supervision half of a run: how to teach, and what counts as an
    intermediate. Both named by the config, independently."""
    return build_method(
        configs.get("method", "coconut"),
        curriculum=configs.get("curriculum"),
        c_thought=configs.get("c_thought", 1),
        max_latent_stage=configs.get("max_latent_stage", 4),
        **_as_dict(configs.get("method_options", {})),
    )


# -- model -------------------------------------------------------------------------

def build_model(configs: Config, tokenizer: STokenizer,
                method: Optional[Method] = None) -> torch.nn.Module:
    """Construct the backbone and wrap it for continuous thoughts.

    ``model_id`` may be a HF model id, a local config json, or absent -- in which case
    a GPT-2 is built from the ``model`` block of the config. Weights are randomly
    initialised (``from_config``): the symbolic vocabulary has no pretrained
    counterpart.

    The wrapper is skipped for text-only methods, which never see a latent token.
    """
    if method is None:
        method = _method_from(configs)
    model_id = configs.get("model_id")
    if model_id:
        base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_id))
    else:
        spec = configs.get("model", Config())
        base = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=spec.get("n_positions", 1024),
                n_embd=spec.get("n_embd", 768),
                n_layer=spec.get("n_layer", 2),
                n_head=spec.get("n_head", 8),
            )
        )
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError(
            f"model vocab_size {base.get_input_embeddings().num_embeddings} != tokenizer "
            f"{len(tokenizer)}. With STokenizer(n_nodes=N) the model needs vocab_size N+9."
        )

    # A GPT2Config carries GPT-2's own eos id (50256), which is not in this symbolic
    # vocabulary at all -- so `generate` would watch for a token that can never appear
    # and always run to max_new_tokens. The text arms then babble past their answer
    # ("[A] 13 [A] 13 ..."), which is scored as a miss.
    for cfg in (base.config, base.generation_config):
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id

    if method.text_only:
        return base  # baselines never see a latent token

    return Coconut(
        base,
        tokenizer.latent_token_id,
        tokenizer.start_latent_id,
        tokenizer.end_latent_id,
        tokenizer.eos_token_id,
        # `loss: set` swaps the intermediate objective for eos's BFS loss. Only
        # meaningful with a set-valued curriculum (ring/ball); a no-op on `path`,
        # whose label sets hold one node.
        loss_kind=configs.get("loss", "ce"),
        # `backprop_last: k` truncates backpropagation to the final k forward passes;
        # null (the default) backpropagates through the whole thought chain.
        backprop_last=configs.get("backprop_last"),
    )


def forward_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Coconut takes positional args and returns a namedtuple-like; a plain HF model
    takes keyword args. Normalising here keeps the loop identical for both."""
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    if isinstance(inner, Coconut):
        return model(
            batch["input_ids"], batch["attention_mask"], batch["labels"],
            batch["position_ids"],
            # Ragged, so the collator keeps it a plain list; only the set loss reads it.
            candidate_nodes=batch.get("candidate_nodes"),
        ).loss
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        position_ids=batch["position_ids"],
    ).loss


# -- curriculum ---------------------------------------------------------------------

def stage_at(epoch: int, configs: Config, method: Method) -> int:
    """Stage as a function of epoch. Methods without a curriculum stay at 0."""
    if method.text_only:
        return 0
    return scheduled_stage(
        epoch,
        epochs_per_stage=configs.epochs_per_stage,
        max_latent_stage=configs.get("max_latent_stage", 4),
        first_stage_epochs=configs.get("first_stage_epochs", 0),
    )


# -- evaluation ----------------------------------------------------------------------

@torch.no_grad()
def _score_single_token(
    inner: "Coconut",
    shard: list[dict[str, Any]],
    tokenizer: STokenizer,
    collate: Collator,
    dist_ctx: Dist,
    batch_size: int,
    samples: Optional[list[dict[str, Any]]],
) -> list[tuple[int, bool, bool, bool]]:
    """One-token scoring in batches -- the fast path behind ``score``.

    Must agree with the per-row path exactly, which turns on reading the argmax at each
    row's OWN last real token. The collator left-pads to align latent blocks and then
    right-pads to the batch width, so that index is neither column 0 nor the last column:
    it is the position of the final ``1`` in the row's attention mask, found by flipping
    the mask and taking the first hit.
    """
    records: list[tuple[int, bool, bool, bool]] = []
    for start in range(0, len(shard), batch_size):
        chunk = shard[start : start + batch_size]
        batch = collate(chunk)
        input_ids = batch["input_ids"].to(dist_ctx.device)
        attention_mask = batch["attention_mask"].to(dist_ctx.device)
        position_ids = batch["position_ids"].to(dist_ctx.device)
        logits = inner(input_ids, attention_mask, None, position_ids).logits
        width = attention_mask.shape[1]
        last = width - 1 - attention_mask.flip(-1).long().argmax(-1)
        chosen = logits[torch.arange(len(chunk), device=logits.device), last].argmax(-1)
        for row, token_id, ids in zip(chunk, chosen.tolist(), input_ids):
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            predicted = text.strip().split()
            expected = row["answer"].split("[A]")[-1].strip().split()
            hit = bool(predicted) and bool(expected) and predicted[0] == expected[0]
            candidates = row.get("candidate_nodes") or []
            set_hit = (None if len(candidates) < 2
                       else bool(predicted) and predicted[0] in {str(c) for c in candidates})
            records.append((row["difficulty"], bool(row.get("is_final", True)), hit, set_hit))
            if samples is not None:
                samples.append({
                    "prompt": tokenizer.decode(ids, skip_special_tokens=True),
                    "expected": " ".join(expected),
                    "predicted": text.strip(),
                    "hit": hit,
                    "difficulty": row["difficulty"],
                    "candidates": [str(c) for c in candidates],
                })
    return records


@torch.no_grad()
def score(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    tokenizer: STokenizer,
    collate: Collator,
    dist_ctx: Dist,
    max_new_tokens: int = 1,
    samples: Optional[list[dict[str, Any]]] = None,
    batch_size: int = 128,
) -> list[tuple[int, bool, bool, bool]]:
    """Generate an answer per prompt and return one ``(difficulty, is_final, hit,
    set_hit)`` record per row, gathered across ranks.

    Each row carries its own expected answer, because ANSWER_ONLY promotes a
    depth-d node to be the target -- it is not always the sample's own ``target``.

    ``set_hit`` is the same judgement widened to the row's candidate set: the emitted
    node had to be *the* sampled label for ``hit``, but only has to be *a* valid
    intermediate for ``set_hit``. With one label drawn per epoch from a frontier of
    size k, a model that has perfectly learned the frontier still scores 1/k on ``hit``
    -- 0.5 on a 2-arm star's ring -- so ``hit`` cannot be thresholded at 0.9 for
    set-valued curricula, and ``set_hit`` can. The two coincide wherever the set has a
    single member, which is every path-curriculum row and every answering row.

    The prediction is taken by *length*, not by splitting the decoded text on ``[A]``:
    intermediate curriculum stages deliberately omit ``[A]`` (they predict the next
    node straight after the latents), so a split would silently return the whole
    prompt and score its first token -- chance-level noise attributed to the model.

    Ranks split the rows and results are gathered.

    ``max_new_tokens == 1`` on a Coconut model takes a BATCHED fast path. That is every
    latent arm's scoring: the answer is one node token, and ``Coconut.generate``'s decode
    loop (``range(max_new_tokens - 1)``) does not execute at all, so "generation" is a
    single ``forward`` plus an argmax -- and ``forward`` already runs batch 128 in
    training. Scoring row by row cost 14.3 ms/row against 0.93 ms/row batched, and eval
    was ~21% of wall-clock (more once every stage is scored), all of it launch overhead on
    an idle GPU. The collator guarantees the invariant this needs: real tokens keep
    positions 0..n-1 whatever padding surrounds them, so a prompt encodes identically at
    any batch size. Text arms (``max_new_tokens > 1``) keep the one-at-a-time path, which
    the wrapper still requires.
    """
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    inner.eval()

    shard = rows[dist_ctx.rank :: dist_ctx.world_size]
    if isinstance(inner, Coconut) and max_new_tokens == 1:
        records = _score_single_token(inner, shard, tokenizer, collate, dist_ctx,
                                      batch_size, samples)
        inner.train()
        return [r for sr in dist_ctx.gather_objects(records) for r in sr]

    records: list[tuple[int, bool, bool, bool]] = []
    for row in shard:
        batch = collate([row])
        input_ids = batch["input_ids"].to(dist_ctx.device)
        attention_mask = batch["attention_mask"].to(dist_ctx.device)
        if isinstance(inner, Coconut):
            output = inner.generate(input_ids, attention_mask, max_new_tokens=max_new_tokens)
        else:
            output = inner.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Both generators return prompt + continuation, and a single-row collate pads
        # to the row's own width, so the continuation starts at the prompt length.
        prompt_len = input_ids.shape[1]
        text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        predicted = text.strip().split()
        # TEXT_COT's stored answer is the whole continuation ("3 14 16 [A] 16"), and
        # its prediction likewise spells out the path, so score the final answer of
        # each. Latent arms have no [A] to split on and fall through unchanged.
        if "[A]" in row["answer"]:
            predicted = text.split("[A]")[-1].strip().split()
        expected = row["answer"].split("[A]")[-1].strip().split()
        hit = bool(predicted) and bool(expected) and predicted[0] == expected[0]
        # Credit any member of the frontier. `None` when the row has no set to widen to,
        # which keeps "this arm has no set-valued labels" distinct from "it has them and
        # they happen to agree with exact match this epoch".
        candidates = row.get("candidate_nodes") or []
        set_hit = (None if len(candidates) < 2
                   else bool(predicted) and predicted[0] in {str(c) for c in candidates})
        records.append((row["difficulty"], bool(row.get("is_final", True)), hit, set_hit))
        if samples is not None:
            # The decoded text only exists here; recovering it later would mean generating
            # twice. Stored raw so the caller decides how many to show and how to format.
            samples.append({
                "prompt": tokenizer.decode(input_ids[0], skip_special_tokens=True),
                "expected": " ".join(expected),
                "predicted": text.strip(),
                "hit": hit,
                "difficulty": row["difficulty"],
                "candidates": [str(c) for c in candidates],
            })

    inner.train()
    return [r for shard_records in dist_ctx.gather_objects(records) for r in shard_records]


def accuracy(records: list[tuple]) -> dict[str, float]:
    """Overall accuracy plus a per-difficulty breakdown.

    ``set_acc`` is reported whenever the rows carry set-valued labels at all -- not merely
    when it currently differs from ``acc``. Keying on the difference made the series vanish
    early in training, exactly when the model predicts outside the frontier and the two
    agree at zero, so a plot of it started mid-run.

    A record may omit ``set_hit`` or set it to ``None``, meaning the row had no set to
    widen to; such rows are excluded from ``set_acc`` rather than counted as agreeing.
    """
    if not records:
        return {}
    widened = [(d, f, hit, (None if len(r) < 4 else r[3]))
               for r in records for d, f, hit in [r[:3]]]
    metrics = {"acc": sum(hit for _, _, hit, _ in widened) / len(widened)}
    by_difficulty: dict[int, list[bool]] = {}
    for difficulty, _, hit, _ in widened:
        by_difficulty.setdefault(difficulty, []).append(hit)
    for difficulty, hits in sorted(by_difficulty.items()):
        metrics[f"acc_d{difficulty}"] = sum(hits) / len(hits)

    with_sets = [(d, s) for d, _, _, s in widened if s is not None]
    if with_sets:
        metrics["set_acc"] = sum(s for _, s in with_sets) / len(with_sets)
        by_difficulty = {}
        for difficulty, set_hit in with_sets:
            by_difficulty.setdefault(difficulty, []).append(set_hit)
        for difficulty, hits in sorted(by_difficulty.items()):
            metrics[f"set_acc_d{difficulty}"] = sum(hits) / len(hits)
    return metrics


def print_samples(samples: list[dict[str, Any]], limit: int, epoch: int,
                  stage: Optional[int] = None) -> None:
    """Show a few of the stage's generations, so a number that looks like chance can be read.

    Aggregate accuracy does not distinguish "emitted the twin", "emitted a node from the
    wrong depth" and "emitted a marker" -- and those have different causes. Several
    conclusions in this project turned on exactly that distinction and needed an offline
    probe to recover it; printing a handful per epoch makes it visible as the run goes.

    Deliberately shows both a hit and a miss when the epoch contains each: a list of
    successes hides the failure mode, and a list of failures hides whether anything works.

    Prints the prompt in full, edge list included. It is the bulk of the line, but it is the
    only way to check the generation against the graph by eye -- whether the emitted node is
    even adjacent to anything on the route, which copy of an isomorphic component it came
    from, and whether the offered candidates are the ones the construction intended.

    The rows are the *current stage's* evaluation rows, matching the gate. Feeding the
    final-task rows here instead was actively misleading: at stage 0 it showed a depth-5
    question with five latents while training was a one-hop question with none, so the
    samples belonged to a format the gate was not measuring.
    """
    if not samples or limit <= 0:
        return
    misses = [x for x in samples if not x["hit"]]
    hits = [x for x in samples if x["hit"]]
    # Interleave so the sample is representative rather than the first N rows, which are in
    # task order and therefore correlated.
    chosen, i = [], 0
    while len(chosen) < limit and (i < len(misses) or i < len(hits)):
        if i < len(misses):
            chosen.append(misses[i])
        if len(chosen) < limit and i < len(hits):
            chosen.append(hits[i])
        i += 1
    where = "" if stage is None else f", stage {stage}"
    print(f"  -- generations at epoch {epoch}{where} "
          f"({len(hits)}/{len(samples)} correct) --", flush=True)
    for x in chosen[:limit]:
        prompt = x["prompt"].replace("<|latent|>", "<L>").replace("<|no-answer|>", "<NA>")
        mark = "OK  " if x["hit"] else "MISS"
        # `candidates` is the *acceptable* answer set (a star's whole BFS ring, say), not the
        # set offered in the prompt -- those are now visible above. Only meaningful when the
        # label is genuinely set-valued, which is what `set_acc` scores.
        extra = ""
        if len(x["candidates"]) > 1:
            extra = "  [acceptable: %s]" % " ".join(x["candidates"][:8])
        print(f"     {mark} d{x['difficulty']}  {prompt}  ->  {x['predicted']!r} "
              f"(want {x['expected']!r}){extra}", flush=True)


def _reachable_arms(row: dict[str, Any], tokenizer: STokenizer):
    """Root, target, and the reachable root->leaf arms parsed from a row's prompt.

    Superposition is a statement about how a thought splits its mass across the *frontier* --
    the nodes reachable at a given depth. On a chain the frontier is one node per depth (one
    arm); on an m-arm star it is m nodes (m arms). Tracing the arms from the root lets the
    lens be read as "how much does thought t back each arm's depth-t node", which is the
    only view that makes superposition legible.

    Parsed from the decoded prompt (edges, [R] root) rather than threaded through as a task,
    so this stays a function of the row alone. Returns None on any parse failure, so a probe
    never crashes training.
    """
    try:
        text = tokenizer.decode(row["input_ids"])
        if "[R]" not in text or "[Q]" not in text:
            return None
        root = int(text.split("[R]")[1].split()[0])
        edge_part = text.split("[Q]")[0].replace(tokenizer.eos_token, "")
        adj: dict[int, list[int]] = {}
        for chunk in edge_part.split("|"):
            parts = chunk.split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                adj.setdefault(u, []).append(v)
        arms: list[list[int]] = []
        def walk(node: int, path: list[int], seen: frozenset) -> None:
            nbrs = [v for v in adj.get(node, []) if v not in seen]
            if not nbrs:
                if path:
                    arms.append(path)
                return
            for v in nbrs:
                walk(v, path + [v], seen | {v})
        walk(root, [], frozenset({root}))
        target = int(str(row.get("answer", "")).split("[A]")[-1].strip())
        return root, target, arms
    except (ValueError, IndexError, KeyError):
        return None


def _print_arms(row: dict[str, Any], tokenizer: STokenizer,
                thought_vecs: dict[int, torch.Tensor], final_top) -> None:
    """One row of the arms view: per reachable arm, the lens probability each thought
    assigns to that arm's depth-matched node, so superposition is readable at a glance.

    A chain has one arm, so one line; a two-arm star has two, and near-equal probabilities
    down the two lines at a given thought is superposition. Falls back to a terse
    thoughts-only line if the prompt cannot be parsed into arms.
    """
    T = max(thought_vecs) if thought_vecs else 0
    parsed = _reachable_arms(row, tokenizer)
    pred = (f"   ->  {final_top[0]}={final_top[1]:.2f}" if final_top else "")
    if not parsed or T == 0:
        # No arms (or no thoughts): show each thought's argmax so the row still says something.
        cells = []
        for t in range(1, T + 1):
            top = torch.topk(thought_vecs[t], 1)
            cells.append(f"t{t}:{tokenizer.decode([int(top.indices[0])])}={float(top.values[0]):.2f}")
        print(f"     (unparsed)  " + "  ".join(cells) + pred, flush=True)
        return
    root, target, arms = parsed
    print(f"     root {root}, target {target}:", flush=True)
    labels = "ABCDEFGH"
    for i, arm in enumerate(arms[:len(labels)]):
        tag = labels[i] + ("*" if target in arm else " ")
        cells = []
        for t in range(1, T + 1):
            if t - 1 < len(arm):
                node = arm[t - 1]
                p = float(thought_vecs[t][node]) if node < thought_vecs[t].numel() else 0.0
                cells.append(f"{node:>2}={p:.2f}")
            else:
                cells.append("      ")
        print(f"       {tag} " + "  ".join(cells) + (pred if i == 0 else ""), flush=True)


@torch.no_grad()
def probe_thoughts(model, rows: list[dict[str, Any]], tokenizer: STokenizer,
                   collate: Collator, dist_ctx: Dist, limit: int,
                   epoch: int, verbose: bool = True,
                   show_rows: int = 10) -> dict[str, float]:
    """Logit lens on every latent: what does each continuous thought decode to?

    Coconut feeds the hidden state at position t back as the input embedding at t+1, so the
    input embedding at a latent position *is* the preceding thought. Projecting it through
    the LM head asks which vocabulary token that thought most resembles -- the same
    computation as scripts/logit_lens.py, run online so the mechanism is visible while the
    curriculum moves rather than only in a post-hoc probe.

    Two numbers matter more than the argmax:

    ``mass_top2``  how much probability the two leading nodes hold between them. High means
        the thought is *about* a small set of nodes rather than diffuse.
    ``split``      min/max of those two. Near 1 is an even split -- two nodes held at once,
        which is superposition. Near 0 is commitment to one.

    Measured offline on a depth-4 star scoring 0.985: thoughts 1-3 sat at split 0.53-0.54
    with 95-98% of mass on two nodes, and thought 4 collapsed to 0.07 -- exactly where the
    two arms first become distinguishable. Logging it per epoch turns that into a curve.

    Returns scalars for the metrics record; prints a per-latent breakdown when ``verbose``.
    """
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(inner, Coconut):
        return {}
    head = getattr(inner.base_causallm, "lm_head", None)
    if head is None:
        return {}

    inner.eval()
    latent_id = tokenizer.latent_token_id
    per_latent: dict[int, list[tuple[float, float]]] = {}
    shown = 0
    for row in rows[:limit]:
        ids = row["input_ids"]
        positions = [i for i, t in enumerate(ids) if t == latent_id]
        if not positions:
            continue
        batch = collate([row])
        out = inner(batch["input_ids"].to(dist_ctx.device),
                    batch["attention_mask"].to(dist_ctx.device),
                    None,
                    batch["position_ids"].to(dist_ctx.device))
        # Re-locate the latents in the *collated* row: the collator left-pads, so the
        # column indices shift and the uncollated ones would silently point at padding.
        collated = batch["input_ids"][0].tolist()
        cols = [i for i, t in enumerate(collated) if t == latent_id]
        # One quantity, not two. The LM head applied to the hidden state at position p is
        # simultaneously (a) the next token the model would emit from p and (b) the content
        # of the thought written at p+1, because Coconut *defines* the latent embedding at
        # p+1 to be that hidden state. Verified numerically: head(inputs_embeds[latent at p])
        # and logits[p-1] agree to 2e-8. Printing both said the same thing twice, shifted by
        # one column.
        #
        # So the lens is read off `logits` at every position, and a latent's content is just
        # the entry belonging to the position before it.
        latent_cols = set(cols)
        first = min([i for i, t in enumerate(collated)
                     if tokenizer.decode([int(t)]) in ("[R]", "[Q]")] or [0])
        rank = 0
        thought_vecs: dict[int, torch.Tensor] = {}   # thought index -> full prob vector
        final_top = None
        for pos in range(first, len(collated)):
            probs = out.logits[0, pos].softmax(-1)
            # This position's outgoing distribution *is* the next thought, when the next
            # position holds one. That is what feeds both the aggregate and the arms view.
            if pos + 1 in latent_cols:
                rank += 1
                top = torch.topk(probs, 2)
                v0, v1 = float(top.values[0]), float(top.values[1])
                per_latent.setdefault(rank, []).append((v0 + v1, v1 / v0 if v0 > 0 else 0.0))
                thought_vecs[rank] = probs
            if pos == len(collated) - 1:                       # the answer position
                top = torch.topk(probs, 2)
                final_top = (tokenizer.decode([int(top.indices[0])]), float(top.values[0]),
                             tokenizer.decode([int(top.indices[1])]), float(top.values[1]))

        if verbose and shown < show_rows:
            if shown == 0:
                print(f"  -- logit lens at epoch {epoch}: probability each thought places on "
                      f"each reachable arm's depth-matched node --", flush=True)
                print(f"     arm* holds the target; superposition = arms near-equal, "
                      f"committed = arm* high and the rest low", flush=True)
            _print_arms(row, tokenizer, thought_vecs, final_top)
        shown += 1
    inner.train()
    if not per_latent:
        return {}

    if verbose:
        print(f"     aggregate over {shown} rows -- mass on the top 2 nodes, and how "
              f"evenly it splits:", flush=True)
        for rank in sorted(per_latent):
            mass = sum(m for m, _ in per_latent[rank]) / len(per_latent[rank])
            split = sum(s for _, s in per_latent[rank]) / len(per_latent[rank])
            # `split` only means anything once the thought is actually *about* two nodes.
            # Early in training mass is ~2/|vocab| and the two values are both tiny, so an
            # even split says "uniform", not "superposition". Requiring mass first stops the
            # label from firing on noise -- it did in the first version of this.
            if mass < 0.5:
                hint = "diffuse (mass too low to read the split)"
            elif split > 0.6:
                hint = "SUPERPOSED"
            elif split < 0.25:
                hint = "committed"
            else:
                hint = "leaning"
            print(f"     thought {rank}: mass {mass:.3f}  split {split:.3f}  {hint}",
                  flush=True)

    masses = [m for v in per_latent.values() for m, _ in v]
    splits = [s for v in per_latent.values() for _, s in v]
    out_metrics = {"thoughts/mass_top2": sum(masses) / len(masses),
                   "thoughts/split": sum(splits) / len(splits)}
    # The first thought is where superposition should be strongest (nothing has forced a
    # choice yet) and the last is where it should collapse, so track those two separately.
    first, last = min(per_latent), max(per_latent)
    out_metrics["thoughts/split_first"] = (
        sum(s for _, s in per_latent[first]) / len(per_latent[first]))
    out_metrics["thoughts/split_last"] = (
        sum(s for _, s in per_latent[last]) / len(per_latent[last]))
    return out_metrics


def evaluate(model, rows, tokenizer, collate, dist_ctx, max_new_tokens: int = 1
             ) -> dict[str, float]:
    """Accuracy over ``rows``, overall and per difficulty."""
    return accuracy(score(model, rows, tokenizer, collate, dist_ctx, max_new_tokens))


def stage_metrics(records: list[tuple]) -> dict[str, float]:
    """Split what a curriculum stage trains into the two skills it can involve.

    ``answer/*``  the label answers the question that was posed -- the same skill
                  eval measures, just on this stage's rows.
    ``step/*``    the label is a waypoint: the next node on the path, with no [A].
                  Only GOLDEN_PATH/FRONTIER train this, and eval never asks for it.

    Averaging the two is what made a single stage_acc uninterpretable -- at stage 3
    of the Coconut curriculum 124 rows are answering while 133 are stepping, so the
    number tracked neither skill. A kind that has no rows is simply absent, which
    keeps each series continuous instead of switching meaning mid-run.
    """
    out: dict[str, float] = {}
    for kind, want in (("answer", True), ("step", False)):
        subset = [r for r in records if r[1] is want]
        if not subset:
            continue
        out[f"{kind}/n"] = float(len(subset))
        out.update({f"{kind}/{k}": v for k, v in accuracy(subset).items()})
    return out


@torch.no_grad()
def validation_loss(model, rows, collate, dist_ctx, batch_size) -> float:
    """Teacher-forced loss on the validation split (legacy's eval/loss)."""
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    inner.eval()
    shard = rows[dist_ctx.rank :: dist_ctx.world_size]
    total, n = 0.0, 0
    for i in range(0, len(shard), batch_size):
        batch = collate(shard[i : i + batch_size])
        # Keep the ragged candidate column: with `loss: set` the objective reads it, and
        # dropping it here would silently score validation under a different loss than
        # training used.
        moved = {k: v.to(dist_ctx.device) for k, v in batch.items() if torch.is_tensor(v)}
        if "candidate_nodes" in batch:
            moved["candidate_nodes"] = batch["candidate_nodes"]
        total += forward_loss(model, moved).item()
        n += 1
    inner.train()
    return dist_ctx.sum_reduce(total / max(n, 1)) / dist_ctx.world_size


# -- training -------------------------------------------------------------------------

def train(configs: Config) -> dict[str, float]:
    dist_ctx = setup_distributed()
    set_seed(configs.seed + dist_ctx.rank)  # decorrelate dropout across ranks

    train_tasks, val_tasks = _tasks_from(configs)
    # Vocabulary follows the data unless pinned. Hand-specifying it means every change
    # of graph family or size needs a matching edit, and getting it wrong surfaces as a
    # device-side index error deep in training rather than at construction.
    n_nodes = configs.get("n_nodes") or 1 + max(
        node for tasks in (train_tasks, val_tasks) for task in tasks
        for edge in task.edges for node in edge
    )
    # Record it, so the saved config.json fully determines the model and any later tool
    # (the latent ablation, a re-score) rebuilds the same vocabulary rather than guessing.
    configs.set("n_nodes", n_nodes)
    # The method is built first because its format decides which structural tokens the
    # vocabulary needs (`[G]` for the question-first layout).
    method = _method_from(configs)
    tokenizer = STokenizer(n_nodes=n_nodes, extra_markers=method.extra_markers)
    collate = Collator(tokenizer, latent_id=tokenizer.latent_token_id)

    model = build_model(configs, tokenizer, method).to(dist_ctx.device)
    # `init_from: path` starts from another run's weights instead of a fresh init. Used to
    # ask whether a circuit learned under one prompt format transfers to another; the
    # optimizer is NOT restored, so this is a re-initialised fine-tune, not a resume.
    init_from = configs.get("init_from")
    if init_from:
        blob = torch.load(init_from, map_location=dist_ctx.device, weights_only=False)
        missing = model.load_state_dict(blob.get("model", blob), strict=False)
        if dist_ctx.is_main:
            print(f"initialised from {init_from}"
                  f"{f' (missing {len(missing.missing_keys)} keys)' if missing.missing_keys else ''}",
                  flush=True)
    if dist_ctx.is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[dist_ctx.local_rank] if torch.cuda.is_available() else None,
            # Every parameter is touched on every pass (the latent count changes how
            # many times a layer runs, not which layers run), so the extra autograd
            # traversal is unnecessary. Exposed in case a future variant needs it.
            find_unused_parameters=configs.get("find_unused_parameters", False),
        )

    def fresh_optimizer() -> torch.optim.Optimizer:
        # `optimizer: sgd` and `lr_schedule: cosine` exist to diagnose the retention failure:
        # a solved checkpoint returns to the guessing floor while its TRAINING loss rises, which
        # is not what continued minimisation of a separable objective does. That points at the
        # update rule rather than the objective, so the update rule has to be variable.
        kind = configs.get("optimizer", "adamw").lower()
        decay = configs.get("weight_decay", 0.01)
        if kind == "sgd":
            return torch.optim.SGD(model.parameters(), lr=configs.lr,
                                   momentum=configs.get("momentum", 0.0),
                                   weight_decay=decay)
        if kind != "adamw":
            raise ValueError(f"unknown optimizer {kind!r}; have adamw, sgd")
        return torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=decay)

    def lr_at(epoch: int) -> float:
        """Learning rate for this epoch, from `lr_schedule` (+ optional `warmup_epochs`)."""
        return lr_schedule_value(
            epoch, peak=configs.lr, total_epochs=configs.num_epochs,
            kind=configs.get("lr_schedule", "constant"),
            warmup=configs.get("warmup_epochs", 0),
            floor=configs.get("lr_min", 1.0e-6),
        )

    optimizer = fresh_optimizer()
    run_dir = Path(configs.get("save_path", "runs")) / configs.name
    if dist_ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(configs.to_dict(), indent=2, default=str))

    logger, log_step = _make_logger(configs, run_dir, dist_ctx)
    history: dict[str, float] = {}
    stage, train_rows, eval_rows = -1, [], []
    # The stage the previous epoch trained on. Starts at the same -1 sentinel as `stage`
    # so epoch 0 still announces its opening stage, as the clock path always did.
    prev_stage: int = -1
    # First epoch/stage at which the benchmark was solved, recorded once and then carried in
    # every row. `eval_example` grants each task its own `depth-1` latents regardless of the
    # training stage, so a curriculum can (and on tree20 does) solve the task many stages
    # before it finishes climbing -- every arm crossed 0.95 at stage 5-6 of 16. Without this
    # the distinction is invisible in the logs and the result reads as "needed all 16 stages".
    solved_threshold = configs.get("solved_threshold", 0.95)
    solved_at: Optional[tuple[int, int]] = None
    # `stop_when_complete`: false runs to `num_epochs` regardless; true stops as soon as the
    # curriculum is complete; an int holds that many further epochs first, which is how you
    # buy evidence that the result is *sustained* rather than a peak -- this project has
    # repeatedly mistaken transient peaks for successes, so the hold is the cautious choice
    # when a number is going into a paper.
    stop_when_complete = configs.get("stop_when_complete", True)
    hold = int(stop_when_complete) if not isinstance(stop_when_complete, bool) else 0
    complete_since: Optional[int] = None
    accumulate = configs.get("gradient_accumulation_steps", 1)
    # `staging: threshold | revisit` drives the curriculum from measured accuracy instead
    # of the clock. A clock advances regardless of whether the stage was learned -- on a
    # length-5 chain Coconut left stage 1 at step accuracy 0.129 because twenty epochs had
    # passed. Empty (the default) keeps the epoch schedule.
    staging_mode = configs.get("staging")
    # 0.98 rather than a looser bar: the gate is exact match on a two-way question whose
    # candidates are shown, so its ceiling is 1.0 and the solved arms sit at 0.99-1.00. A
    # threshold well below that ceiling admits a stage the model has merely half-learned,
    # and under `stop_when_complete` the run then stops there -- measured, an arm gated at
    # 0.9 finished the ladder at epoch 49 and scored 0.918 on the test split, against 0.998
    # for the same rule that took 103 epochs.
    staging_threshold = configs.get("staging_threshold", 0.98)
    # Two independent brakes; see `next_stage` for the distinction.
    #
    # `staging_patience` = consecutive passes required, a noise filter.
    # `staging_min_epochs` = epochs of training required at a stage before the curriculum may
    #     pass it, a coverage guarantee. It exists because a stage can pass its first probe
    #     purely by generalisation from shallower ones -- probing stage 12 succeeds on an arm
    #     that never trained there -- and the scan would then run past several stages having
    #     trained none of them. At 1, every stage gets an epoch of its own training.
    #
    # Defaults are 1 and 1: requiring training is the weaker condition, and it is the one
    # actually wanted. Patience 2 would additionally hold a stage that failed, was trained
    # once, and then passed -- which is exactly the case where advancing is correct.
    staging_patience = configs.get("staging_patience", 1)
    staging_min_epochs = configs.get("staging_min_epochs", 1)
    streaks: dict[int, int] = {}
    # stage -> epochs trained at it, accumulated over the run (a stage returned to keeps its
    # earlier epochs, since what matters is whether it was ever taught, not recency).
    trained_epochs: dict[int, int] = {}
    if staging_mode:
        stage = 0

    # Record the RESOLVED staging knobs, then re-write the snapshot. config.json is dumped
    # before this block runs, so a run that relies on the defaults recorded `null` for its
    # threshold and patience -- and the only way to recover them afterwards was to infer the
    # gate from the accuracies at which stages were promoted. Anything read through
    # `.get(key, default)` and used to steer training belongs in the snapshot.
    for key, value in (("staging_threshold", staging_threshold),
                       ("staging_patience", staging_patience),
                       ("staging_min_epochs", staging_min_epochs),
                       ("max_grad_norm", configs.get("max_grad_norm", 1.0)),
                       ("save_every", configs.get("save_every", 50)),
                       ("uniform_prob", configs.get("uniform_prob", 0.0)),
                       ("log_all_stage_acc", configs.get("log_all_stage_acc", True))):
        configs.set(key, value)
    if dist_ctx.is_main:
        (run_dir / "config.json").write_text(
            json.dumps(configs.to_dict(), indent=2, default=str))

    for epoch in range(configs.get("resume", 0), configs.num_epochs):
        # `stage_changed` must compare against the stage TRAINED LAST EPOCH, not against
        # `stage` itself. Under adaptive staging `next_stage` writes `stage` at the *end* of
        # the epoch, so `new_stage` is already that value here and `new_stage != stage` was
        # comparing a variable with itself -- permanently False. That silently made
        # `reset_optimizer: "stage"` a no-op for every `revisit`/`threshold` run (the whole
        # resetStage axis of a tree20 factorial was a duplicate of resetF) and suppressed
        # the `[stage N]` transition print. `prev_stage` is the stage this loop actually
        # trained on last time round, which is the only thing a transition can be measured
        # against. (`should_reset_optimizer` separately ignores epoch 0, so announcing the
        # opening stage does not cause a spurious reset.)
        new_stage = (stage if staging_mode else stage_at(epoch, configs, method))
        # Rebuild every epoch, not just on a stage change. Each build re-draws which
        # node is taken from a multi-node label set, which tasks get mixed to an
        # easier stage (uniform_prob), and the edge/candidate order -- so a frozen
        # dataset is a much smaller effective training set. Caching it across a stage
        # cost ~24 accuracy points against the legacy stack.
        stage_changed = new_stage != prev_stage
        prev_stage = stage = new_stage
        # This epoch trains at `stage`, so the gate below may count it. Incremented here
        # rather than after training, because the gate runs inside the same iteration and the
        # epoch's training has completed by then.
        trained_epochs[stage] = trained_epochs.get(stage, 0) + 1
        # `resample_each_epoch: false` freezes the data: every epoch then rebuilds
        # byte-identical rows -- same edge order, same candidate order and position, same
        # label draw from a set-valued curriculum, same per-example stage, same batch
        # composition. The training set stops being resampled and becomes a fixed corpus.
        #
        # Measured consequence, on ATC-style arms where the label is one of the shown
        # candidates: the label sits in a CONSTANT slot for 200/200 tasks, against 3/200
        # (chance) when the epoch is added. That is a memorisation shortcut -- "for this
        # graph the answer is the first candidate" -- which needs no reasoning and which
        # `shuffle_nodes` does not prevent, since that re-permutes ids and not positions.
        # The comment in build_training_set records that caching rows across a stage cost
        # ~24 accuracy points against the reference implementation.
        # `hidden_anneal_epochs: N` STAGES the prompt format instead of mixing it at a fixed
        # rate: candidates start hidden and are revealed linearly over N epochs. Mixing a small
        # constant fraction (3%, 10%) did not escape, while a model pre-trained entirely without
        # candidates transfers at 0.913 zero-shot and fine-tunes to stage 5 -- so what works is
        # having the circuit before candidates arrive, which argues for a schedule rather than
        # a blend.
        anneal = configs.get("hidden_anneal_epochs")
        if anneal:
            start = configs.get("hidden_prob_start", 1.0)
            method.options["hidden_prob"] = start * max(0.0, 1.0 - epoch / max(anneal, 1))
            record_hidden = method.options["hidden_prob"]
        else:
            record_hidden = method.options.get("hidden_prob", 0.0)
        epoch_offset = epoch if configs.get("resample_each_epoch", True) else 0
        shared = dict(
            seed=configs.seed + epoch_offset,
            perturb=configs.get("shuffle_nodes", False),
        )
        train_rows = build_training_set(
            train_tasks, method, tokenizer, stage=stage,
            uniform_prob=configs.get("uniform_prob", 0.0),
            # Gate which *tasks* are admitted by difficulty, a separate knob from the
            # stage schedule and from `curriculum` (which states are intermediates).
            # max(stage, 1): no task has difficulty 0, so admitting "<= 0" empties the
            # split and the sampler dies on num_samples=0.
            max_difficulty=max(stage, 1) if configs.get("grow_task_difficulty", False)
            else None,
            max_edges=_edge_budget(stage, configs),
            **shared,
        )
        # Two different questions, deliberately kept apart:
        #
        # <method>/*  "is the curriculum working?" -- can the model do what THIS stage
        #             trains? Stage-dependent, and it means something different per
        #             method: for Coconut the next node on the path, for ATC an easier
        #             depth-(stage+1) question. A training diagnostic, and not
        #             comparable across arms -- hence the per-method namespace.
        # eval/*      "can it solve the task?" -- every task at its own full difficulty
        #             and full latent budget. Fixed for the whole run and identical for
        #             every method, so arms are comparable. This is the reported number.
        task_rows = build_eval_set(val_tasks, method, tokenizer, **shared)
        stage_rows = build_eval_set(val_tasks, method, tokenizer, stage=stage, **shared)
        # Teacher-forced loss on the val split, matching the training distribution.
        val_loss_rows = build_training_set(
            val_tasks, method, tokenizer, stage=stage,
            uniform_prob=configs.get("uniform_prob", 0.0), shuffle=False, **shared,
        )
        if should_reset_optimizer(configs.get("reset_optimizer", False), epoch, stage_changed):
            optimizer = fresh_optimizer()
        epoch_lr = lr_at(epoch)
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        if dist_ctx.is_main and stage_changed:
            print(f"[stage {stage}] {len(train_rows)} train rows, {len(task_rows)} eval rows",
                  flush=True)

        sampler = (
            DistributedSampler(train_rows, shuffle=True, seed=epoch, drop_last=True)
            if dist_ctx.is_distributed else None
        )
        loader = DataLoader(
            train_rows,
            batch_size=configs.batch_size_training,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate,
            num_workers=configs.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=dist_ctx.is_distributed,  # keep step counts equal across ranks
        )
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        started, total_loss, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            # Ragged non-tensor columns (candidate_nodes) stay on the host: the set loss
            # indexes with them rather than computing on them.
            candidates = batch.get("candidate_nodes")
            batch = {k: v.to(dist_ctx.device) for k, v in batch.items() if torch.is_tensor(v)}
            if candidates is not None:
                batch["candidate_nodes"] = candidates
            is_update = (step + 1) % accumulate == 0 or step + 1 == len(loader)
            # Skip the gradient sync on accumulation steps.
            sync = nullcontext() if (is_update or not dist_ctx.is_distributed) else model.no_sync()
            with sync:
                loss = forward_loss(model, batch) / accumulate
                loss.backward()
            if is_update:
                # Clip by default. The reference symbolic setup clips at 1.0, and the failure
                # we most want to rule out looks like a divergence event rather than a slow
                # decay: a truncated-backprop arm went from train loss 1.09 to 2.95 across a
                # single stage transition and never recovered. Set
                # `max_grad_norm: 0` to disable.
                #
                # Placement is load-bearing. This runs inside `is_update`, i.e. after the
                # update step's `backward()` has completed, so DDP has already all-reduced the
                # gradients and the norm computed here is the true global one. Under a sharded
                # scheme (FSDP) the same call would see only the local shard and silently clip
                # against the wrong norm -- use the wrapper's own `clip_grad_norm_` there.
                #
                # NOTE for comparability: every result recorded before this default changed ran
                # with NO clipping, including tree20's 0.999. Re-runs are not drop-in
                # comparable with those numbers.
                clip = configs.get("max_grad_norm", 1.0)
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step_loss = loss.item() * accumulate
            total_loss += step_loss
            n_batches += 1
            # Legacy logs train metrics per batch, so its wandb x-axis is
            # steps-not-epochs; match that so the two stacks overlay directly.
            log_step({"train/epoch": epoch + 1,
                      "train/step": epoch * len(loader) + step,
                      "train/loss": step_loss})

        mean_loss = dist_ctx.sum_reduce(total_loss / max(n_batches, 1)) / dist_ctx.world_size
        record = {"epoch": epoch + 1, "stage": stage, "train/epoch_loss": mean_loss,
                  "train/hidden_prob": record_hidden, "train/lr": epoch_lr,
                  "time_s": round(time.time() - started, 1)}

        if (epoch + 1) % configs.get("eval_every", 1) == 0 or epoch + 1 == configs.num_epochs:
            # Latent arms need one token (the node after the thoughts); the text
            # baselines must decode a whole answer span.
            max_new = method.max_new_tokens
            record.update({f"eval/{k}": v for k, v in
                           accuracy(score(model, task_rows, tokenizer, collate, dist_ctx,
                                          max_new_tokens=max_new)).items()})
            if solved_at is None and record.get("eval/acc", 0.0) >= solved_threshold:
                solved_at = (epoch + 1, stage)
                if dist_ctx.is_main:
                    print(f"[solved] eval/acc {record['eval/acc']:.3f} >= {solved_threshold} "
                          f"at epoch {epoch + 1}, stage {stage} of {configs.get('max_latent_stage', 4)}",
                          flush=True)
            # `show_thoughts: N` runs the logit lens over N benchmark rows. Scalars go into
            # the record so superposition is a curve rather than a one-off probe.
            n_probe = configs.get("show_thoughts", 0)
            if n_probe:
                record.update(probe_thoughts(
                    model, task_rows, tokenizer, collate, dist_ctx, n_probe, epoch + 1,
                    verbose=dist_ctx.is_main,
                    show_rows=configs.get("show_thought_rows", 10)))
            # The curriculum diagnostic goes under the method's own panel: what each
            # stage trains differs per method, so these series are not comparable
            # across arms and should not share an axis with anyone else's. Only
            # eval/* is common ground.
            #
            # `show_samples: N` prints N of THESE generations (0 disables) -- the current
            # stage's question at the current stage's latent budget, which is what the gate
            # scores and what the curriculum acts on. Printing the final-task rows instead
            # showed a five-latent depth-5 question while stage 0 was being trained on a
            # one-hop question with no latents, so a gate stuck at chance came with samples
            # that could not explain it.
            n_show = configs.get("show_samples", 0)
            stage_samples: list[dict[str, Any]] = [] if n_show else None
            record.update({f"{method.label}/{k}": v for k, v in
                           stage_metrics(score(model, stage_rows, tokenizer, collate,
                                               dist_ctx, max_new_tokens=max_new,
                                               samples=stage_samples)).items()})
            if n_show and dist_ctx.is_main:
                print_samples(stage_samples, n_show, epoch + 1, stage=stage)
            # The gate and the benchmark must be read on FIXED formats, never on the training
            # mixture: while `hidden_prob` anneals, a rising gate would otherwise conflate the
            # model improving with the mixture getting easier. Both are logged every epoch.
            if configs.get("hidden_anneal_epochs") or configs.get("hidden_prob"):
                was = method.options.get("hidden_prob", 0.0)
                for tag, prob in (("visible", 0.0), ("hidden", 1.0)):
                    method.options["hidden_prob"] = prob
                    rows = build_eval_set(val_tasks, method, tokenizer, stage=stage, **shared)
                    kinds = stage_metrics(score(model, rows, tokenizer, collate, dist_ctx,
                                                max_new_tokens=max_new))
                    for key, value in kinds.items():
                        record[f"fixed/{tag}/{key}"] = value
                method.options["hidden_prob"] = was
            record["eval/loss"] = validation_loss(
                model, val_loss_rows, collate, dist_ctx, configs.batch_size_training
            )

        def _edge_of(candidate: int) -> Optional[int]:
            """The difficulty ceiling in force at ``candidate``, or None when not growing."""
            if not configs.get("grow_task_difficulty", False):
                return None
            return max(candidate, 1)

        if solved_at is not None:
            record["stage/solved_at_epoch"], record["stage/solved_at_stage"] = solved_at
        if staging_mode:
            # Score a stage at ITS OWN budget and question, never the current stage's, so
            # the number means the same thing every time it is measured. Gate on the
            # weakest kind present: at a handover stage some tasks are answering while
            # deeper ones still step, and mastering one is not mastering the stage.
            def accuracy_at(candidate: int) -> float:
                # Score only the difficulties this stage actually trains on. With
                # `grow_task_difficulty` the training set holds difficulty <= stage while the
                # validation set spans every difficulty, so gating on the whole split marks
                # the model on material it has not been shown -- which made the
                # length-curriculum arms read 0.165 against a 0.25 chance floor.
                rows = build_eval_set(val_tasks, method, tokenizer, stage=candidate,
                                      max_difficulty=_edge_of(candidate), **shared)
                kinds = stage_metrics(score(model, rows, tokenizer, collate, dist_ctx,
                                            max_new_tokens=method.max_new_tokens))
                # Which number gates the curriculum is a deliberate choice, not something
                # to infer from what happens to be logged, because the two have different
                # ranges and a threshold is only meaningful against one of them:
                #
                #   acc      exact match against the one sampled label. With a ring of
                #            size k this CAPS at 1/k however well the frontier was learned
                #            -- 0.5 on a 2-arm star -- while chance is 1/|vocab| = 0.045,
                #            since the prompt shows the final pair and the node must be
                #            generated. Threshold near 0.45 is the calibrated choice.
                #   set_acc  credit for naming any valid frontier member. Ranges over the
                #            full 0..1, so a 0.9 threshold means what it usually means.
                #
                # Defaulting to set_acc silently would make a 0.45 threshold trivially
                # passable, so the metric is named in config and only falls back when the
                # requested one is absent (path arms log no set_acc at all).
                want = configs.get("staging_metric", "acc")
                accs = []
                for kind in ("answer", "step"):
                    if f"{kind}/acc" not in kinds:
                        continue
                    key = f"{kind}/{want}"
                    accs.append(kinds[key] if key in kinds else kinds[f"{kind}/acc"])
                return min(accs) if accs else 0.0

            stage, measured, streaks = next_stage(
                stage, accuracy_at=accuracy_at, threshold=staging_threshold,
                max_stage=configs.get("max_latent_stage", 4), mode=staging_mode,
                streaks=streaks, patience=staging_patience,
                trained=trained_epochs, min_epochs=staging_min_epochs,
            )
            # Completion must be judged on what the SCAN reached, before the logging backfill
            # widens `measured`. The scan only walks past a stage that passes with a full
            # streak, so `max_stage` appearing here with a passing score means every stage
            # below it was reliable too -- that is the whole curriculum, done.
            #
            # Reading it off `stage` and `streaks` alone was wrong once `threshold` gained its
            # monotone clamp: with a regression at, say, stage 3, the scan stops at 3 and never
            # touches `max_stage`, yet the clamp still reports `stage == max_stage` and
            # `streaks[max_stage]` keeps a stale value from when the stage really was mastered
            # -- so a broken run would announce itself complete.
            _top = configs.get("max_latent_stage", 4)
            # The top stage must also have been trained, or a run whose scan reached the top
            # purely by generalisation would declare itself complete having never taught it.
            scan_finished = (measured.get(_top, -1.0) >= staging_threshold
                             and streaks.get(_top, 0) >= staging_patience
                             and trained_epochs.get(_top, 0) >= staging_min_epochs)
            # `log_all_stage_acc` backfills the stages the staging RULE did not need to
            # measure, purely for the plot -- the decision above is already made and is
            # left untouched. `threshold` (monotone) only scores the current stage, so
            # earlier stages drop out of wandb exactly when you want to watch them for
            # forgetting; `revisit` stops scanning at the first unlearned stage, so higher
            # stages are missing too. This scores every remaining stage 0..max_stage at its
            # own budget so `stage/acc_s{k}` is a complete, comparable series under either
            # mode.
            #
            # ON BY DEFAULT. It was opt-in, and the cost of that was a result we could not
            # diagnose: a truncated-backprop arm under monotone staging collapsed from 0.98
            # to chance at a stage transition, and because monotone scores only the current
            # stage, the early-stage decay that preceded the collapse was never recorded.
            # The extra eval per unmeasured stage is worth paying to never be blind to that
            # again. Set `log_all_stage_acc: false` to opt out when eval cost dominates
            # (it is one extra scored stage per epoch per unmeasured stage, so the price
            # grows with `max_latent_stage` -- worst case under `threshold`, where the rule
            # itself measures only one).
            if configs.get("log_all_stage_acc", True):
                for candidate in range(configs.get("max_latent_stage", 4) + 1):
                    if candidate not in measured:
                        measured[candidate] = accuracy_at(candidate)
            record["stage/next"] = stage
            record.update({f"stage/acc_s{k}": v for k, v in measured.items()})
            record.update({f"stage/streak_s{k}": v for k, v in streaks.items()})
            # Curriculum complete: at the deepest stage, and the scan confirmed every stage
            # up to it is reliably passing this epoch (see `scan_finished` above).
            #
            # Without this a solved run burns the rest of `num_epochs` at a stage it has
            # already mastered: the tree20 arms completed around epoch 100-105 and then ran
            # ~40 more epochs for ~0.001 of accuracy.
            complete = stage >= _top and scan_finished
            record["stage/complete"] = int(complete)
        logger(record)
        history = record

        if dist_ctx.is_main:
            inner = model.module if isinstance(model, DistributedDataParallel) else model
            # checkpoint_latest.pt holds the full training state (weights, optimizer,
            # position in the curriculum, latest metrics) and is overwritten each
            # epoch, so a run is always resumable and always re-scorable without
            # keeping 300 copies -- upstream saves every epoch and its four arms
            # reached 138 GB. save_every additionally keeps epoch-stamped snapshots
            # when the training trajectory itself is wanted -- every 50 epochs by
            # default, at ~180 MB each, so a 300-epoch run keeps six. Set 0 to disable.
            save_every = configs.get("save_every", 50)
            if save_every and (epoch + 1) % save_every == 0:
                torch.save(inner.state_dict(), run_dir / f"checkpoint_{epoch + 1}.pt")
            torch.save({"epoch": epoch + 1, "stage": stage,
                        "model": inner.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": record}, run_dir / "checkpoint_latest.pt")
        dist_ctx.barrier()

        if stop_when_complete and record.get("stage/complete"):
            if complete_since is None:
                complete_since = epoch
            if epoch - complete_since >= hold:
                if dist_ctx.is_main:
                    inner = (model.module if isinstance(model, DistributedDataParallel)
                             else model)
                    torch.save({"epoch": epoch + 1, "stage": stage,
                                "model": inner.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "metrics": record}, run_dir / "checkpoint_complete.pt")
                    print(f"[complete] curriculum finished at stage {stage} "
                          f"(streak {streaks.get(stage, 0)} >= {staging_patience}), "
                          f"eval/acc {record.get('eval/acc', float('nan')):.3f}; "
                          f"stopping at epoch {epoch + 1} of {configs.num_epochs} and "
                          f"saving checkpoint_complete.pt", flush=True)
                dist_ctx.barrier()
                break
        else:
            complete_since = None

    if dist_ctx.is_distributed:
        dist.destroy_process_group()
    return history


def _make_logger(configs: Config, run_dir: Path, dist_ctx: Dist):
    """Print on rank 0, mirror to metrics.jsonl, and log to wandb when enabled."""
    method_label = _method_from(configs).label
    if not dist_ctx.is_main:
        return (lambda record: None), (lambda record: None)

    wandb_run = None
    if configs.get("wandb", False):
        import wandb

        wandb_run = wandb.init(project=configs.get("project", "lyot-curriculum"), name=configs.name,
                               config=configs.to_dict())
    metrics_path = run_dir / "metrics.jsonl"

    def log_step(record: dict[str, Any]) -> None:
        """Per-batch metrics: wandb only. Mirroring these to metrics.jsonl would
        write one line per gradient step."""
        if wandb_run:
            wandb_run.log(record)

    def log(record: dict[str, Any]) -> None:
        parts = [f"epoch {record['epoch']:>4}", f"stage {record['stage']}",
                 f"loss {record['train/epoch_loss']:.4f}"]
        # What this stage trains, then the fixed benchmark. A method whose stages only
        # ever answer (ATC, the baselines) prints one train figure; Coconut prints
        # both, and during its stage-3 handover prints them side by side.
        for kind, label in (("step", "step"), ("answer", "train")):
            key = f"{method_label}/{kind}/acc"
            if key in record:
                parts.append(f"{label} {record[key]:.3f}")
        if "eval/acc" in record:
            parts.append(f"task {record['eval/acc']:.3f}")
        if "eval/loss" in record:
            parts.append(f"eval_loss {record['eval/loss']:.3f}")
        print(" | ".join(parts) + f" | {record['time_s']}s", flush=True)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if wandb_run:
            wandb_run.log(record)

    return log, log_step


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a continuous-thought curriculum")
    parser.add_argument("config_file")
    parser.add_argument("overrides", nargs="*", help="key=value, e.g. lr=3e-4 model.n_layer=4")
    args = parser.parse_args(argv)
    configs = Config.from_yaml(args.config_file, args.overrides)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(configs, flush=True)
    train(configs)


if __name__ == "__main__":
    main()
