# Learn Your Own Thoughts

Continuous chain-of-thought learned from **answers alone**, by a curriculum over task
*difficulty* rather than over intermediate steps.

A latent reasoning token carries a hidden state rather than a symbol, so it can hold a whole
search frontier at once instead of committing to one branch. The difficulty is training it: with
no written trace there is no target for the latent, and existing methods supply one either by
converting a discrete chain (Coconut) or by supervising a designated intermediate state such as a
BFS frontier node. Both require someone to decide what the latents should represent, and both then
need a purpose-built metric to decide when a stage has been learned.

ATC supplies neither. At stage `s` the model is asked a depth-`min(s+1, d)` instance of the *same*
task with `min(s, d-1)` latent tokens, and only the answer is supervised. Because the objective at
every stage is the task, the curriculum advances on ordinary validation accuracy — nothing about
the latents is inspected or decoded.

## The task

Each instance is a random tree on 20 nodes plus a disjoint isomorphic copy: 40 nodes and 38 edges,
edges listed in shuffled order, a root, and two candidate nodes. One candidate is reachable from
the root; the distractor is its image under the isomorphism, so the two agree on depth, in-degree
and out-degree and no candidate-local statistic separates them. Difficulty is the hop distance from
root to target, 3–16 with mean 8.3. Chance is 50%.

Graphs are generated, not shipped: train, validation and test come from disjoint RNG streams
(`seed`, `seed + 10_000`, `seed + 20_000`), so a split is reconstructible from a run's own
`config.json`.

## Install

```bash
uv sync                       # or: pip install -e . -r requirements-lock.txt
```

Needs one GPU per arm; every number below was measured on a single H100 80GB in float32.

## Reproduce

Each arm is one command. Runs write to `runs/<name>/` — `metrics.jsonl` per epoch,
`checkpoint_latest.pt` every epoch, and `checkpoint_complete.pt` when the curriculum finishes.

```bash
# ATC, answer-only supervision — the reported arm
python -m lyot.run configs/tree20/atc.yaml \
    staging=threshold staging_threshold=0.9 staging_patience=5 staging_min_epochs=0

# baselines
python -m lyot.run configs/tree20/no_cot.yaml
python -m lyot.run configs/tree20/cot.yaml
python -m lyot.run configs/tree20/coconut_setacc.yaml
python -m lyot.run configs/tree20/coconut_clock.yaml lr=1.0e-4 epochs_per_stage=25 num_epochs=600

# controls: the curriculum with one ingredient removed
python -m lyot.run configs/tree20/no_curriculum.yaml     # full depth and budget from epoch 1
python -m lyot.run configs/tree20/data_curriculum.yaml   # difficulty grows, no latents at all
python -m lyot.run configs/tree20/max_budget.yaml        # difficulty grows, budget fixed at max
```

`atc.yaml` deliberately omits the staging knobs so they follow the code defaults (threshold 0.98,
patience 1, minimum dwell 1). The overrides above restore the gate the reported number was measured
with; without them the same recipe finishes sooner and scores 98.5.

Then score a finished run on the held-out split — this is how every number below was produced:

```bash
python scripts/score_test_split.py runs/<name> \
    --checkpoint checkpoint_complete.pt --n-test 5000
```

It rebuilds the model and vocabulary from the run's own `config.json`, regenerates the test split,
and writes `test_eval_<checkpoint>.json` with overall and per-difficulty accuracy.

## Results

Held-out test split, 5,000 graphs, full backpropagation. Chance is 50.0.

| Method | Intermediate supervision | Stage criterion | Acc. (%) |
|---|---|---|---|
| No-CoT | No | — | 48.3 |
| Explicit CoT | Yes | — | 63.5 |
| No curriculum | No | — | 47.8 |
| Data-only curriculum | No | task accuracy | 49.0 |
| Coconut | Yes | fixed epochs per stage | 63.0 |
| Coconut | Yes | set accuracy ≥ τ | 95.6 |
| **ATC** | **No** | task accuracy ≥ τ | **99.8** |

Two things the controls establish. Sixteen latents available from the first epoch, with the
curriculum removed, score 47.8 — indistinguishable from having no latents at all. Keeping the
difficulty ordering but removing the latents scores 49.0. Neither ingredient works alone.

Coconut reaches 95.6, but only once gated on *set accuracy* over its forced intermediate
prediction — a metric that exists only because the intermediates are supervised. ATC gates on the
number it reports.

### Scaling: truncated backpropagation

A depth-`d` instance unrolls `d-1` latent passes, so backpropagating through all of them ties the
cost of a step to the depth reached. `backprop_last=b` runs all but the last `b` passes under
`no_grad`, cutting both the recurrence and the KV cache, which makes the backward cost `O(b)`.

|  | full backward | truncated, `b=5` |
|---|---|---|
| advance-only staging | 99.8 | 51.0 |
| backtracking staging | **99.8** | **97.4** |
| peak memory (GiB) | 5.78 | 1.55 (**0.27×**) |
| time per step (ms) | 198.6 | 124.7 (**0.63×**) |

With full gradients the staging rule is irrelevant; with a truncated window it decides between a
working model and chance. Full backpropagation was implicitly maintaining the earlier difficulties,
and once that is removed only an explicit repair — falling back to the shallowest difficulty that
regressed — recovers it.

```bash
# the four cells above
python -m lyot.run configs/tree20/atc.yaml staging=revisit   \
    staging_threshold=0.9 staging_patience=5 staging_min_epochs=0 [backprop_last=5]
python -m lyot.run configs/tree20/atc.yaml staging=threshold \
    staging_threshold=0.9 staging_patience=5 staging_min_epochs=0 [backprop_last=5]

# cost table
python scripts/bench_backward.py configs/tree20/atc.yaml --stage 16 --windows full,8,5,4,2,1
```

## Layout

```
lyot/
  tasks/        graph families and the twin-graph reachability instance
  curriculum.py which states count as intermediates: path, ring (BFS frontier), ball
  methods/      what to ask and what to label: atc, coconut, cot, no-cot
  collator.py   left-pads so latent positions align across a batch
  coconut.py    the latent wrapper: hidden state at t becomes the input at t+1
  schedule.py   the accuracy gate, and assembling an epoch's rows
  run.py        training loop, evaluation, staging, checkpointing
configs/tree20/ one YAML per arm; any key is overridable as key=value on the CLI
scripts/        score_test_split.py, bench_backward.py
```

Three knobs carry most of the design:

| knob | what it does |
|---|---|
| `staging` | `threshold` advances on the current difficulty only; `revisit` consults every difficulty and falls back to the shallowest that regressed; `revisit-monotone` detects without repairing |
| `backprop_last` | backward window `b`; `null` is full backpropagation |
| `method_options.budget` | `stage` grants `s` latents, `depth-1` grants `min(s, d-1)` (the fencepost), `max` grants the final-stage budget from the start |

## License

MIT, see `LICENSE`.
