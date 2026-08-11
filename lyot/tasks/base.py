"""The Task interface.

A ``Task`` is one problem instance. It must be able to render itself as a prompt and
say what the correct answer is; everything else is optional structure that particular
methods may ask for.

Adding a new problem family means implementing ``Task`` (and a loader or generator) --
no change to any method, collator, or training loop.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence


class Task(ABC):
    """One problem instance.

    ``difficulty`` is the number of reasoning steps a correct solution needs. It sets
    the curriculum's upper bound for this instance and labels the metric buckets, so
    it must be the *true* minimum -- not an upper bound or a guess.
    """

    @property
    @abstractmethod
    def difficulty(self) -> int:
        """Reasoning steps required. 0 means directly answerable."""

    @abstractmethod
    def prompt(self, rng: random.Random) -> str:
        """The question, as wire-format text. May randomise presentation order."""

    @property
    @abstractmethod
    def answer(self) -> str:
        """The correct response to ``prompt``."""

    def perturbed(self, rng: random.Random) -> "Task":
        """An equivalent instance with surface details resampled.

        Defaults to no-op. Graph tasks override it to permute node ids, which stops
        the model learning id statistics instead of structure (eos's ``shuffle_nodes``).
        """
        return self


@dataclass(frozen=True)
class TaskSet:
    """A named collection of tasks -- a split.

    Carries the name so logs and errors can say which split they mean, and because a
    generated set has no file path to fall back on.
    """

    name: str
    tasks: Sequence[Task]

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def __getitem__(self, index: int) -> Task:
        return self.tasks[index]

    @property
    def max_difficulty(self) -> int:
        return max((task.difficulty for task in self.tasks), default=0)


# Registry so configs can name a task family as a string.
_LOADERS: dict[str, Callable[..., Sequence[Task]]] = {}


def register_loader(name: str, loader: Callable[..., Sequence[Task]]) -> None:
    _LOADERS[name] = loader


def load_tasks(path: str | Path, *, family: str = "reachability",
               limit: Optional[int] = None, name: Optional[str] = None) -> TaskSet:
    """Load a split from disk.

    Files hold raw problem instances. Anything derivable -- shortest paths, BFS rings,
    per-depth node sets -- is computed on demand by whichever method wants it, never
    stored, so one file serves every method.
    """
    if family not in _LOADERS:
        raise KeyError(f"unknown task family {family!r}; have {sorted(_LOADERS)}")
    with open(path) as handle:
        raw: list[dict[str, Any]] = json.load(handle)
    if limit:
        raw = raw[:limit]
    return TaskSet(name=name or Path(path).stem, tasks=_LOADERS[family](raw))
