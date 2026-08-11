"""The Method interface.

A method answers three questions for a given task at a given curriculum stage:
what to ask, what to label, and how many latents to allow. Everything a method needs
about the problem it derives from the ``Task``; nothing is read from precomputed data.

Which states count as valid intermediates is a separate axis -- see ``lyot.curriculum``
-- injected here rather than hardcoded, so any method composes with any strategy.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from lyot.tasks import Task
from lyot.curriculum import DEFAULT as DEFAULT_CURRICULUM
from lyot.curriculum import Curriculum
from lyot.curriculum import build as build_curriculum

LATENT = "<|latent|>"
NO_ANSWER = "<|no-answer|>"
EOS = "<eos>"


@dataclass(frozen=True)
class Example:
    """One built training example, before tokenization."""

    prompt: str
    target: str
    difficulty: int = 0
    # Does the label answer the question that was posed, or is it a waypoint on the
    # way there? Coconut's early stages predict the next node on the path, which is a
    # different skill from producing an answer; averaging the two hides both.
    is_final: bool = True
    # Set of acceptable labels, when a method's objective is set-valued (Coconut-ring).
    candidates: tuple[int, ...] = ()


class Method(ABC):
    """How to teach a task.

    Args:
        curriculum: which states count as valid intermediates at a given depth. An
            orthogonal choice -- a method decides how an intermediate is *used*, not
            which ones qualify -- so it is injected rather than baked in.
        c_thought: latent tokens emitted per reasoning step.
        max_latent_stage: the deepest curriculum stage; also the latent ceiling.
    """

    #: registry key, and the namespace its metrics are logged under
    name: str = ""
    #: True when the method never uses latent tokens (the text baselines)
    text_only: bool = False

    def __init__(self, *, curriculum: Optional[Curriculum] = None, c_thought: int = 1,
                 max_latent_stage: int = 4, **kwargs):
        self.curriculum = curriculum or build_curriculum()
        self.c_thought = c_thought
        self.max_latent_stage = max_latent_stage
        self.options = kwargs

    def intermediates(self, task: Task, depth: int) -> Sequence[int]:
        """Valid states after ``depth`` steps, per the configured strategy."""
        return self.curriculum.at(task, depth)

    def stage_for(self, task: Task, scheduled_stage: int) -> int:
        """Clamp the global stage to what this task can support."""
        return min(scheduled_stage, task.difficulty)

    def latents(self, count: int) -> str:
        """``count`` reasoning steps rendered as latent tokens, with spacing."""
        return " " + f"{LATENT} " * (count * self.c_thought) if count else " "

    def close(self, prompt: str, latents: str, mark: bool = True) -> str:
        """Assemble prompt + thoughts, then ``[A]`` when ``mark``.

        Deliberately independent of ``is_final``. Whether a row *answers* the question posed
        is a semantic question -- ATC's stages each pose a legitimate easier question, so its
        rows always answer one -- while whether ``[A]`` appears turns out to be a mechanical
        one about token adjacency, and the two do not coincide.

        Why it matters, from the bridge factorial. Holding labels fixed and toggling only the
        marker: ``none/promoted`` scored 1.000 without it and 0.057 with it; ``both/final``
        (Coconut) scored 1.000 without and 0.061 with. A plausible mechanism is adjacency --
        Coconut's design has the label immediately follow the last latent, so that thought's
        hidden state predicts the label directly. Inserting ``[A]`` puts a token in between,
        whose embedding is identical at every stage, and the thought must route through it.

        Not a universal rule, though: ``both/promoted`` failed either way (0.540 with, 0.277
        without), so candidates-shown-on-a-promoted-question fails for a separate reason.

        Both branches normalise the join. The tokenizer splits on whitespace, so a missing
        space would turn ``8[A]`` into a single out-of-vocabulary token rather than two.
        """
        body = f"{prompt}{latents}".rstrip()
        return f"{body} [A]" if mark else body + " "

    @abstractmethod
    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        """Build the training example this method wants at ``stage``."""

    def eval_example(self, task: Task, rng: random.Random) -> Example:
        """The benchmark question: the real task at its full latent budget.

        Identical across methods and across curricula by construction --
        this is the only thing that makes arms comparable -- so it must not consult
        ``self.curriculum``.
        """
        budget = min(task.difficulty, self.max_latent_stage)
        latents = "" if self.text_only else self.latents(budget)
        # The benchmark must be posed the same way training was, or the two measure
        # different things -- in particular how many candidates are shown has to apply
        # to both. `hide_candidates` is the older spelling of `show="none"`.
        show = self.options.get("show", "both")
        if self.options.get("hide_candidates", False):
            show = "none"
        prompt = task.prompt(rng, show=show,
                             n_candidates=self.options.get("n_candidates"),
                             anchor_mark=self.options.get("anchor_mark", "[R]"),
                             layout=self.options.get("layout", "default"),
                             filler=self.options.get("filler", ""),
                             repeat_anchor=self.options.get("repeat_anchor", False),
                             graph_mark=self.options.get("graph_mark", "[G]"))
        # ...and so does the answer marker. Appending [A] here unconditionally while training
        # with answer_mark=False poses the benchmark in a format the model never saw, which
        # drives eval BELOW its own chance floor -- measured 0.03 against a floor of 0.5,
        # while the stage gate (which uses the training format) sat at 0.5 as expected.
        mark = self.options.get("answer_mark", True)
        return Example(self.close(prompt, latents, mark), task.answer, task.difficulty)

    @property
    def extra_markers(self) -> tuple[str, ...]:
        """Structural tokens this method's format needs beyond the nine fixed ones.

        Declared by the method and passed to ``STokenizer`` by ``run.py``, so a format and
        its vocabulary cannot fall out of step. An undeclared marker is not a silent
        placeholder -- tokenization raises on it -- but it would raise mid-run rather than
        at construction.
        """
        extra = []
        if self.options.get("layout", "default") in ("question-first", "root-first"):
            extra.append(self.options.get("graph_mark", "[G]"))
        if self.options.get("filler", False):
            from lyot.tasks.reachability import FILLER
            extra.append(FILLER)
        return tuple(extra)

    @property
    def max_new_tokens(self) -> int:
        """Generation budget at eval. Latent arms emit one node; text arms a span."""
        return 64 if self.text_only else 1

    @property
    def label(self) -> str:
        """Name for logs: the method, plus the curriculum when they are not the default."""
        if self.text_only or self.curriculum.name == DEFAULT_CURRICULUM:
            return self.name
        return f"{self.name}-{self.curriculum.name}"


_METHODS: dict[str, type[Method]] = {}


def register_method(cls: type[Method]) -> type[Method]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name")
    _METHODS[cls.name] = cls
    return cls


def get_method(name: str) -> type[Method]:
    if name not in _METHODS:
        raise KeyError(f"unknown method {name!r}; have {sorted(_METHODS)}")
    return _METHODS[name]


def build(name: str, *, curriculum: Optional[str] = None, **kwargs) -> Method:
    """Assemble a method with its curriculum -- the two axes of a run's supervision,
    chosen independently by config."""
    return get_method(name)(curriculum=build_curriculum(curriculum), **kwargs)
