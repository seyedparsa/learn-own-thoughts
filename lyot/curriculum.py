"""Curriculum: which states count as legitimate intermediates at depth *d*.

The third axis of a run, alongside the task and the method. A method decides *how* an
intermediate is used -- Coconut labels it as the next step, ATC promotes it to be the
answer -- but not *which* states qualify. That is this choice, and the two vary
independently:

    method: coconut   curriculum: path    upstream Coconut (rbs)
    method: coconut   curriculum: ring    Coconut-BFS / superposition (eos)
    method: atc       curriculum: path    ATC as run so far
    method: atc       curriculum: ring    a curriculum over frontiers, not yet tried

Upstream expressed the path/ring difference by shipping two corpora of the same
graphs. Both sets are derivable from the graph, so here it is a config word.

Ordering note: every strategy returns sorted node ids, so a seeded ``rng.choice``
over the result is reproducible regardless of how a data file happened to store them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from lyot.tasks import Task


class Curriculum(ABC):
    """A rule for enumerating valid depth-``d`` intermediate states of a task."""

    #: registry key, and the suffix that distinguishes a non-default arm in logs
    name: str = ""

    @abstractmethod
    def at(self, task: Task, depth: int) -> Sequence[int]:
        """The states a solver could legitimately occupy after ``depth`` steps."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PathCurriculum(Curriculum):
    """Only states on a shortest path to the answer.

    The supervision upstream's ``*_coconut.json`` files encode: a near-deterministic
    target for each step (mean 1.2 nodes on ProsQA), so a single-label loss is well
    posed and intermediate accuracy can approach 1.
    """

    name = "path"

    def at(self, task: Task, depth: int) -> Sequence[int]:
        return _require(task, "path_nodes")(depth)


class RingCurriculum(Curriculum):
    """Every state at exactly ``depth`` steps -- the search frontier.

    The superposition setting: what a parallel breadth-first search holds after
    ``depth`` expansions. Much wider than the path (mean 3.4 nodes on ProsQA), so a
    loss that samples one member caps accuracy near E[1/|ring|] = 0.40 -- ``Example``
    carries the whole set in ``candidates`` for a set-valued objective to use.
    """

    name = "ring"

    def at(self, task: Task, depth: int) -> Sequence[int]:
        return _require(task, "ring")(depth)


class BallCurriculum(Curriculum):
    """Every state within ``depth`` steps -- the cumulative reachable set.

    What the eos *valid* file stores (its train file stores rings instead), and the
    widest of the three: each depth is a superset of the last.
    """

    name = "ball"

    def at(self, task: Task, depth: int) -> Sequence[int]:
        return _require(task, "ball")(depth)


def _require(task: Task, query: str):
    method = getattr(task, query, None)
    if method is None:
        raise TypeError(
            f"{type(task).__name__} does not support {query!r} curriculum; "
            f"implement {query}(depth) on the task or choose another strategy"
        )
    return method


_CURRICULA: dict[str, type[Curriculum]] = {
    strategy.name: strategy for strategy in (PathCurriculum, RingCurriculum, BallCurriculum)
}

#: what a config gets when it does not say. Reproduces upstream Coconut and ATC.
DEFAULT = "path"


def build(name: str | None = None) -> Curriculum:
    name = name or DEFAULT
    if name not in _CURRICULA:
        raise KeyError(f"unknown curriculum {name!r}; have {sorted(_CURRICULA)}")
    return _CURRICULA[name]()
