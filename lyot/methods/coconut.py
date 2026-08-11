"""Coconut: supervise the reasoning path one step at a time.

Stage ``s`` gives ``s`` latents and labels the state reached after ``s+1`` steps, so
the model learns to make each latent carry one step. The question never changes -- it
is always the real one -- and only the deepest stage carries ``[A]``.

Which states are valid at that depth is the ``curriculum`` axis, not part of this
method:

    method: coconut  curriculum: path   upstream Coconut -- one node toward the answer
    method: coconut  curriculum: ring   Coconut-BFS -- any node on the search frontier

Upstream shipped those as two data corpora. With a wide strategy the label is sampled
from a set the prompt cannot disambiguate, so a single-label loss is capped near
E[1/|set|]; ``Example.candidates`` carries the whole set for a set-valued objective.
"""

from __future__ import annotations

import random

from lyot.methods.base import Example, Method, register_method
from lyot.tasks import Task


@register_method
class Coconut(Method):
    """Step-by-step supervision toward the answer."""

    name = "coconut"

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        latents = self.latents(stage)
        if stage >= task.difficulty:
            # Deepest stage: the real question, answered.
            return Example(f"{task.prompt(rng)}{latents}[A]", task.answer,
                           task.difficulty, is_final=True)

        # Draw the prompt before the label: both consume the rng, and this is the
        # order upstream uses, so a shared seed reproduces its exact examples.
        prompt = task.prompt(rng)
        options = self.intermediates(task, stage + 1)
        label = rng.choice(options)
        # No [A]: an intermediate step predicts the next state straight after the
        # latents, so the marker stays reserved for answering the actual question.
        return Example(
            f"{prompt}{latents}".rstrip() + " ", str(label),
            stage + 1, is_final=False, candidates=tuple(options),
        )
