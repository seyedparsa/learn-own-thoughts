"""The bridge between Coconut and ATC, as three independent switches.

Coconut solves a depth-10 chain (0.999) and ``atc`` with ``show=none`` sits at 0.5 on the
same task with the *same labels at every stage*. Only three things separate them, and this
module makes each one a config flag so the 2x2x2 can be run:

    candidates   both | none    are the two candidate nodes shown, or placeholders?
    answer_mark  on   | off     does an intermediate row end with [A]?
    asks         final | promoted
                                which pair is shown: the task's real target pair, fixed for
                                the whole run (Coconut), or the promoted intermediate and its
                                twin, re-posed at every stage (ATC)?

    coconut  == (both, off, final)
    atc-none == (none, on,  promoted)

``asks`` is the axis that is easy to miss and probably the important one. Coconut's ``[Q]``
never changes: it always names the final target pair, so it is a stable anchor identifying
*which component* to search while the label walks outward. ATC re-poses the question each
stage, so the anchor moves with the label. With ``candidates=none`` there is no anchor at
all -- and the measured behaviour of ``atc-none`` is exactly that: at stage 0 it emits the
correct depth-1 node 60/150 times and the *other component's* depth-1 node 90/150, with zero
predictions anywhere else. It has learned "a depth-1 node" and not "from this root".

Labels are identical to Coconut's in every cell: the depth-(stage+1) node on the path, or the
task's answer at the deepest stage. Nothing here changes what is supervised, only how the
question is presented.
"""

from __future__ import annotations

import random

from lyot.methods.base import Example, Method, register_method
from lyot.tasks import Task


@register_method
class BridgeCoconut(Method):
    """Coconut-to-ATC, with the three differences exposed as options.

    Args:
        candidates: ``both`` shows the pair, ``none`` replaces them with placeholders.
        answer_mark: append ``[A]`` to intermediate rows. Coconut omits it, reserving the
            marker for rows that actually answer; ATC appends it always, so the model sees
            ``[A]`` followed by a non-final node.
        asks: ``final`` keeps ``[Q]`` on the task's own target pair at every stage;
            ``promoted`` re-poses it around the current intermediate and its twin.
    """

    name = "bridge"

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        candidates = self.options.get("candidates", "both")
        answer_mark = self.options.get("answer_mark", False)
        asks = self.options.get("asks", "final")
        latents = self.latents(stage)

        if stage >= task.difficulty:
            # Deepest stage: the real question, answered. Identical in every cell, and
            # identical to Coconut -- this is the row the benchmark measures.
            prompt = task.prompt(rng, show=candidates)
            return Example(f"{prompt}{latents}[A]", task.answer,
                           task.difficulty, is_final=True)

        # Prompt before label: both draw from the rng, and this order reproduces
        # Coconut's exact examples under a shared seed.
        options = self.intermediates(task, stage + 1)
        if asks == "final":
            prompt = task.prompt(rng, show=candidates)
            label = rng.choice(options)
        else:
            # Draw the label first, because it decides which pair is shown.
            label = rng.choice(options)
            prompt = task.prompt(rng, reachable=label, show=candidates)

        body = f"{prompt}{latents}"
        text = f"{body}[A]" if answer_mark else body.rstrip() + " "
        return Example(text, str(label), stage + 1,
                       is_final=False, candidates=tuple(options))

    def eval_example(self, task: Task, rng: random.Random) -> Example:
        """The benchmark, posed the way this cell poses questions.

        ``candidates`` has to carry over -- an arm trained without candidates and scored
        with them would be measured on a question it never saw. ``asks`` and
        ``answer_mark`` do not apply: at full depth the promoted node *is* the target, and
        an answering row always ends in ``[A]``.
        """
        budget = min(task.difficulty, self.max_latent_stage)
        prompt = task.prompt(rng, show=self.options.get("candidates", "both"))
        return Example(f"{prompt}{self.latents(budget)}[A]", task.answer, task.difficulty)
