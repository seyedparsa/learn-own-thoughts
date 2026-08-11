"""Scheduling and dataset assembly: when difficulty rises, and how rows are built.

Sits between tasks and methods. Decides which stage each example is built at, asks the
method to build it, and tokenizes the result. Knows nothing about graphs and nothing
about supervision.

Distinct from ``lyot.curriculum``, which names the *other* sense of the word: this
module owns the schedule over epochs, that one owns which states count as valid
intermediates at a given depth.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from transformers import PreTrainedTokenizerBase

from lyot.methods import Example, Method
from lyot.tasks import Task, TaskSet

IGNORE_INDEX = -100


def scheduled_stage(epoch: int, *, epochs_per_stage: int, max_latent_stage: int,
                    first_stage_epochs: int = 0) -> int:
    """Stage as a function of epoch, capped.

    Upstream let this grow unbounded and relied on per-sample clamping, which makes
    the logged stage misleading once it passes the deepest task.

    ``epochs_per_stage=0`` means *no curriculum*: spend no epochs at any stage, i.e. start
    at ``max_latent_stage`` and stay there. That is the control the curriculum is measured
    against -- the full-difficulty question at the full latent budget from the first step --
    and it needs to be expressible without a second code path.
    """
    if epochs_per_stage == 0:
        return max_latent_stage
    if epoch < first_stage_epochs:
        return 0
    advanced = (epoch - first_stage_epochs) // max(epochs_per_stage, 1)
    return min(advanced, max_latent_stage)


def sample_stage(task: Task, scheduled: int, rng: random.Random,
                 uniform_prob: float = 0.0) -> int:
    """This example's stage: the scheduled one, or occasionally an easier one.

    Mixing earlier stages back in (upstream's ``uniform_prob``) stops them being
    forgotten as the curriculum advances.
    """
    top = min(scheduled, task.difficulty)
    return rng.randint(0, top) if uniform_prob and rng.random() < uniform_prob else top


def encode(example: Example, tokenizer: PreTrainedTokenizerBase, *,
           index: Optional[int] = None, with_labels: bool = True) -> dict[str, Any]:
    """Tokenize an Example into the fields the collator expects."""
    prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
    row: dict[str, Any] = {
        "input_ids": prompt_ids,
        "attention_mask": [1] * len(prompt_ids),
        "position_ids": list(range(len(prompt_ids))),
    }
    if with_labels:
        target_ids = tokenizer.encode(example.target, add_special_tokens=False)
        ids = prompt_ids + target_ids
        row = {
            "input_ids": ids,
            "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
            "attention_mask": [1] * len(ids),
            "position_ids": list(range(len(ids))),
        }
    row["difficulty"] = example.difficulty
    row["is_final"] = example.is_final
    # The expected answer travels with the row: ATC promotes an intermediate node to
    # be the target, so it is not always the task's own answer.
    row["answer"] = example.target
    if example.candidates:
        row["candidate_nodes"] = list(example.candidates)
    if index is not None:
        row["idx"] = index
    return row


def build_training_set(
    tasks: TaskSet,
    method: Method,
    tokenizer: PreTrainedTokenizerBase,
    *,
    stage: int,
    seed: int = 0,
    uniform_prob: float = 0.0,
    perturb: bool = False,
    max_difficulty: Optional[int] = None,
    max_edges: Optional[int] = None,
    shuffle: bool = True,
) -> list[dict[str, Any]]:
    """Build one epoch of training rows.

    Deterministic given ``seed`` and rank-independent, so every rank builds the same
    data locally instead of rank 0 broadcasting it.

    Rebuild this every epoch, not once per curriculum stage: each call re-draws which
    node is taken from a multi-node label set, which examples get mixed down by
    ``uniform_prob``, and the edge and candidate order. Caching it across a stage cost
    24 accuracy points against the reference implementation.

    Args:
        max_difficulty: drop tasks needing more steps than this -- the data curriculum,
            which introduces harder instances as training progresses.
        max_edges: drop tasks with more edges than this -- a curriculum over *search
            breadth* rather than depth. Measured: with answer-only supervision a
            one-hop lookup is learned at 0.997 in an 8-edge graph and 0.53 in a 12-edge
            one, the task being otherwise identical, so breadth is the binding
            constraint and nothing else we vary touches it.
    """
    rng = random.Random(seed)
    rows = []
    # One relabelling per drawn graph, shared by its re-rootings. See `_relabel`.
    relabellings: dict[Any, dict[int, int]] = {}
    for index, task in enumerate(tasks):
        if max_difficulty is not None and task.difficulty > max_difficulty:
            continue
        if max_edges is not None and len(task.edges) > max_edges:
            continue
        if perturb:
            task = _relabel(task, rng, relabellings)
        at = sample_stage(task, stage, rng, uniform_prob)
        rows.append(encode(method.example(task, at, rng), tokenizer, index=index))
    if shuffle:
        rng.shuffle(rows)
    return rows


def build_eval_set(
    tasks: TaskSet,
    method: Method,
    tokenizer: PreTrainedTokenizerBase,
    *,
    seed: int = 0,
    stage: Optional[int] = None,
    perturb: bool = False,
    max_difficulty: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Prompts only, in task order, each tagged with its source index.

    With ``stage=None`` (the default) this is the *benchmark*: every task at its own
    full difficulty and full latent budget, identical for every method and fixed for
    the whole run. That is the only measurement comparable across arms.

    With a ``stage`` it is the curriculum diagnostic instead: what this method trains
    right now, which means something different for each of them.
    """
    rng = random.Random(seed)
    rows = []
    relabellings: dict[Any, dict[int, int]] = {}
    # Dropout (`hidden_prob`) is a training-time augmentation. Validation -- the staging gate
    # included -- must see the un-dropped distribution, or a dropout run would be gated on
    # the hidden rows it never means to be scored on. Zero it for the duration of this build.
    opts = getattr(method, "options", None)
    saved_hidden = opts.get("hidden_prob") if isinstance(opts, dict) else None
    if saved_hidden:
        opts["hidden_prob"] = 0.0
    try:
        for index, task in enumerate(tasks):
            if max_difficulty is not None and task.difficulty > max_difficulty:
                continue
            if perturb:
                task = _relabel(task, rng, relabellings)
            example = (method.eval_example(task, rng) if stage is None
                       else method.example(task, min(stage, task.difficulty), rng))
            rows.append(encode(example, tokenizer, index=index, with_labels=False))
    finally:
        if saved_hidden:
            opts["hidden_prob"] = saved_hidden
    return rows


def _relabel(task: Task, rng: random.Random,
             cache: dict[Any, dict[int, int]]) -> Task:
    """Permute a task's node ids, sharing one permutation across a graph's re-rootings.

    ``roots="all"`` emits one task per component of the same graph, differing only in the
    root token and the label. Relabelling each of them independently turns them back into
    unrelated graphs and silently removes the whole point of the option -- measured directly:
    the four variants came out with different edge lists, so the intervention was inert and
    its arms were indistinguishable from the one-root arms.

    Tasks carrying no ``graph`` id (file-loaded, or ``roots="one"``) are relabelled
    independently, which is the old behaviour.
    """
    key = task.payload.get("graph") if getattr(task, "payload", None) else None
    if key is None:
        return task.perturbed(rng)
    mapping = cache.get(key)
    if mapping is None:
        mapping = cache[key] = task.relabelling(rng)
    elif not all(node in mapping for edge in task.edges for node in edge):
        # Belt and braces: a shared permutation must cover this task's nodes. If a future id
        # scheme ever collides across graphs of different sizes, relabel independently rather
        # than raise -- the counterfactual pairing is lost, but only for the colliding group.
        return task.perturbed(rng)
    return task.perturbed(rng, mapping)


def next_stage(current: int, *, accuracy_at, threshold: float, max_stage: int,
               mode: str = "threshold", streaks: Optional[dict[int, int]] = None,
               patience: int = 1, trained: Optional[dict[int, int]] = None,
               min_epochs: int = 0) -> tuple[int, dict[int, float], dict[int, int]]:
    """Pick the next curriculum stage from measured accuracy rather than the clock.

    A clock schedule advances on ``epoch // epochs_per_stage`` regardless of whether the
    stage was learned. Measured on a length-5 chain, Coconut left stage 1 with a step
    accuracy of 0.129 simply because twenty epochs had elapsed, and never recovered.

    ``accuracy_at(k)`` must score stage ``k`` **at stage k's own budget and question**, not
    the current stage's, so a stage means the same thing every time it is measured and the
    numbers are comparable across epochs.

    Two independent brakes, which are easy to confuse:

    ``patience`` is how many *consecutive passes* a stage needs before the curriculum moves
    beyond it. It is a noise filter. A single passing epoch is noisy: with patience 1 a
    revisit run spent 200 of 288 epochs at stage 0 and only two or three at stages 4 and 5,
    leaping forward and snapping back almost every epoch. ``streaks`` maps stage ->
    consecutive passes so far, carried between epochs and returned updated; a stage's streak
    resets the moment it fails.

    ``min_epochs`` is how many epochs must have been *trained* at a stage before it may be
    passed, counted in ``trained``. It is not a noise filter but a coverage guarantee, and it
    exists because a stage can pass its very first probe without ever having been trained:
    depth generalises, so probing stage 12 succeeds on an arm whose curriculum never reached
    it, and the scan would then run straight past stages 6-11 having trained none of them.
    With ``min_epochs=1`` every stage gets at least one epoch of its own training before the
    curriculum leaves it.

    The two differ in the case that matters: a stage that fails, is trained for one epoch,
    and then passes. ``min_epochs=1`` is satisfied and the curriculum advances;
    ``patience=2`` would hold it for a second passing epoch. Requiring training is the
    weaker, cheaper condition, so the run loop's defaults are ``patience=1, min_epochs=1``.
    Here ``min_epochs`` defaults to 0 -- the rule needs a caller that counts training epochs,
    and without ``trained`` every stage would look untrained and nothing could ever advance.

    Modes:
        ``threshold``  advance by one once the current stage has passed ``patience`` epochs
            running. Monotone, so a stage forgotten later is never revisited.
        ``revisit``    scan stages in order and stop at the first that is not yet reliably
            passing -- because it fails now, because its streak is still short, or because it
            has not been trained. Failing sends the curriculum back immediately, since
            catching forgetting the moment it appears is the point.

    Returns ``(next_stage, measured_accuracies, streaks)``.
    """
    if mode not in ("threshold", "revisit", "revisit-monotone"):
        raise ValueError("staging must be 'threshold', 'revisit' or 'revisit-monotone', "
                         f"got {mode!r}")
    streaks = dict(streaks or {})
    trained = trained or {}

    def untrained(stage: int) -> bool:
        return trained.get(stage, 0) < min_epochs

    if mode == "threshold":
        # Advance-only, and therefore consults ONLY the current stage. A rule that cannot
        # return to an earlier stage has no business being gated by one: it could see the
        # regression and do nothing about it, which in practice throttles the curriculum and
        # then stalls it. Measured on tree20: `acc_s0` drifts in a 0.82-0.98 band because
        # training is at stage 5-7 and rehearsal gives stage 0 only ~1.4% of examples, so a
        # gate at 0.90 halted the scan on roughly a third of epochs and reset the streak,
        # stretching one promotion to 20-40 epochs. Gating on retention without repair is
        # `revisit-monotone` below, kept as an explicit control rather than the default
        # meaning of "advance when this stage is learned".
        measured = {current: accuracy_at(current)}
        if measured[current] < threshold:
            streaks[current] = 0
            return current, measured, streaks
        streaks[current] = streaks.get(current, 0) + 1
        if (streaks[current] >= patience and not untrained(current)
                and current < max_stage):
            return current + 1, measured, streaks
        return current, measured, streaks

    # `revisit` and `revisit-monotone` share one scan: walk up from stage 0 and stop at the
    # shallowest stage that is not yet reliably passing -- it fails now, it passes but has not
    # done so `patience` times running, or it has never been trained. Falling out of the loop
    # means every stage is reliable and trained, so the curriculum is finished and the answer
    # is `max_stage`.
    measured: dict[int, float] = {}
    picked = max_stage
    for candidate in range(max_stage + 1):
        measured[candidate] = accuracy_at(candidate)
        if measured[candidate] < threshold:
            streaks[candidate] = 0
            picked = candidate
            break
        streaks[candidate] = streaks.get(candidate, 0) + 1
        if streaks[candidate] < patience:
            # Passing, but not yet reliably. Stay and let the streak build rather than
            # leaping ahead on a single good epoch.
            picked = candidate
            break
        if untrained(candidate):
            # Passing and reliable, but only by generalisation from shallower stages -- the
            # curriculum has never actually trained here. Stop and do so.
            picked = candidate
            break

    # The two scan modes differ in exactly one thing: whether the stage may decrease.
    #
    #   revisit           take the scan's answer as it stands, so a regression below the
    #                     frontier sends training back to repair it at once.
    #   revisit-monotone  clamp to the current stage. It still *detects* a regression -- the
    #                     scan stops there, so nothing above is reached and the frontier
    #                     cannot advance -- but it never retrains it.
    #
    # `revisit-monotone` is the clean control for the fallback: identical ascent rule,
    # identical patience, repair removed. Expect it to be *worse* than either alternative --
    # detection it cannot act on stalls the curriculum -- which is the point of having it.
    if mode == "revisit-monotone":
        picked = max(picked, current)
    return picked, measured, streaks
