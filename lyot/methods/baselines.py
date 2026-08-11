"""Text baselines: no latent tokens, no curriculum.

Both decode a span rather than a single node, so both targets end with ``<eos>``.
Without it generation runs to the full token budget and degenerates ("[A] 13 [A] 13
..."), which scored 26/60 against 46/60 for the same generations cut at the first
marker. Upstream terminates both too (rbs dataset.py:351 and :427).

Neither has a curriculum, so ``stage`` is ignored and the training example *is* the
eval example -- which is why their train and eval accuracies coincide exactly, a
useful consistency check on the metric plumbing.
"""

from __future__ import annotations

import random

from lyot.methods.base import EOS, Example, Method, register_method
from lyot.tasks import Task


@register_method
class TextCoT(Method):
    """Write the reasoning path out as tokens, then the answer."""

    name = "cot"
    text_only = True

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        path = " ".join(str(node) for node in task.shortest_path(rng))
        return Example(task.prompt(rng), f"{path} [A] {task.answer} {EOS}",
                       task.difficulty)

    def eval_example(self, task: Task, rng: random.Random) -> Example:
        return self.example(task, 0, rng)


@register_method
class NoCoT(Method):
    """Answer directly, with no intermediate reasoning."""

    name = "no-cot"
    text_only = True

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        return Example(f"{task.prompt(rng)} [A]", f"{task.answer} {EOS}", task.difficulty)

    def eval_example(self, task: Task, rng: random.Random) -> Example:
        return self.example(task, 0, rng)
