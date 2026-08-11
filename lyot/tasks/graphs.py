"""Graph families: how the structure of a reachability instance is built.

A family answers one question -- what does the graph look like? -- and returns a plain
directed edge list plus the node the search starts from. Everything else (which node is
the target, which is the distractor, how it renders, what counts as an intermediate) is
the job of ``ReachabilityTask``, the curriculum, and the method.

Two-component construction
--------------------------
Every family here builds *one* component and the task duplicates it, so the distractor
is an exact isomorphic twin of the target sitting in an unreachable copy. Measured over
2436 (target, distractor) pairs from random 20-node trees: identical in-degree,
out-degree, and depth-within-component in 2436/2436 cases.

That matters because it is the property ProsQA lacks. There the distractor is
unreachable *and* structurally distinct (257/257), so "which of these two is reachable"
can be answered from local features -- and ATC's latents duly turned out inert. A twin
distractor cannot be told apart without actually tracing connectivity from the start.

Adding a family
---------------
Write a function that returns ``Component`` and register it. It needs no knowledge of
tasks, curricula, or methods; ``lyot.tasks.generate`` handles the rest.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

try:
    import networkx as nx
except ImportError:  # pragma: no cover - only the random-tree family needs it
    nx = None


@dataclass(frozen=True)
class Component:
    """One connected structure, before it is duplicated into a full instance.

    Args:
        edges: directed ``(u, v)`` pairs over nodes ``0..n_nodes-1``.
        n_nodes: node count; ids must be dense so the copy can be offset by it.
        start: where the search begins.
        depth: hops from ``start`` to each reachable node.
        spine: an optional sequence of nodes to draw targets from, ordered from the
            start outward. Families that want difficulty to sweep one chain (tree,
            star) supply the chain; the flower supplies its hubs, which sit several
            hops apart. Use ``targets()`` rather than indexing it -- position along
            the spine is not the same as depth.
    """

    edges: Sequence[tuple[int, int]]
    n_nodes: int
    start: int
    depth: dict[int, int]
    spine: Optional[Sequence[int]] = None

    @property
    def max_depth(self) -> int:
        return max(self.depth.values(), default=0)

    def nodes_at(self, depth: int) -> list[int]:
        return sorted(node for node, d in self.depth.items() if d == depth)

    def targets(self, which: str = "deepest") -> list[tuple[int, int]]:
        """``(difficulty, node)`` pairs to build tasks from, easiest first.

        ``deepest`` (the default) returns only the far end of the spine, so one graph
        yields one task and that task *is* the full problem. Easier versions of it are
        then the curriculum's business -- Coconut supervises the nodes along the way,
        ATC promotes one of them to be the answer -- rather than something the
        generator bakes into the data. Two mechanisms producing easier problems is one
        too many.

        ``all`` returns every spine depth, which is the data-curriculum setting: a
        genuinely easier *instance*, not a subtask of a harder one. Gate it with
        ``grow_task_difficulty`` if you want difficulty admitted over time.

        Difficulty is the node's true BFS depth, never its index along the spine --
        the flower's hubs sit three hops apart, so the two disagree. Reading it from
        ``depth`` keeps "difficulty d" meaning "d hops" for every family.
        """
        spine = self.spine if self.spine is not None else sorted(self.depth, key=self.depth.get)
        pairs = [(self.depth[node], node) for node in spine if self.depth.get(node, 0) > 0]
        if which == "deepest":
            return pairs[-1:]
        if which == "all":
            return pairs
        raise ValueError(f"targets must be 'deepest' or 'all', got {which!r}")


def _depths(edges: Sequence[tuple[int, int]], start: int) -> dict[int, int]:
    """BFS depth from ``start``; unreachable nodes are simply absent."""
    adjacency: dict[int, list[int]] = {}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
    depth = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency.get(node, ()):
            if neighbour not in depth:
                depth[neighbour] = depth[node] + 1
                queue.append(neighbour)
    return depth


# -- families ------------------------------------------------------------------------

def random_tree(rng: random.Random, *, n: int = 100, **_) -> Component:
    """Uniform random labelled tree, rooted at 0, edges oriented parent to child.

    Depth grows like sqrt(n), so n=100 reaches difficulty ~10-20 -- far past ProsQA's
    3-4 hops, which is the range where a latent budget has room to matter.
    """
    if nx is None:
        raise ImportError("random_tree needs networkx; pip install networkx")
    undirected = nx.random_labeled_tree(n, seed=rng.randint(0, 2**31 - 1))
    edges: list[tuple[int, int]] = []
    depth = {0: 0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbour in undirected.neighbors(node):
            if neighbour not in depth:
                depth[neighbour] = depth[node] + 1
                edges.append((node, neighbour))
                queue.append(neighbour)

    deepest = max(depth.values())
    leaf = rng.choice([n_ for n_, d in depth.items() if d == deepest])
    parent = {v: u for u, v in edges}
    spine, node = [leaf], leaf
    while node != 0:
        node = parent[node]
        spine.append(node)
    spine.reverse()
    return Component(edges, n, 0, depth, spine)


def star(rng: random.Random, *, arms: int = 10, length: int = 10, **_) -> Component:
    """``arms`` disjoint chains of ``length`` hanging off a shared centre.

    Every (depth, arm) pair is a distinct node, so the correct answer at each step is
    unique -- which makes it the clean setting for probing what a latent encodes.
    """
    edges: list[tuple[int, int]] = []
    for arm in range(arms):
        base = 1 + arm * length
        edges.append((0, base))
        edges.extend((base + i, base + i + 1) for i in range(length - 1))
    n_nodes = 1 + arms * length
    chosen = rng.randrange(arms)
    base = 1 + chosen * length
    spine = [0] + [base + i for i in range(length)]
    return Component(edges, n_nodes, 0, _depths(edges, 0), spine)


def flower(rng: random.Random, *, petals: int = 5, short: int = 3, long: int = 5,
           dead: int = 3, **_) -> Component:
    """Chained petals, each offering a short route, a long route, and a dead end.

    Adjacent petals share a node -- the hub of one is the start of the next -- so
    solving it means committing to a route repeatedly. The dead ends punish a model
    that expands blindly rather than tracking which branch reaches the hub.
    """
    petal_size = short + long + dead
    stride = petal_size - 1
    template: list[tuple[int, int]] = []

    template.extend((i, i + 1) for i in range(short - 1))
    if short > 0:
        template.append((short - 1, petal_size - 1))

    long_nodes = list(range(short, short + long - 1))
    if long_nodes:
        template.append((0, long_nodes[-1]))
        template.extend((long_nodes[i], long_nodes[i - 1])
                        for i in range(len(long_nodes) - 1, 0, -1))
        template.append((long_nodes[0], petal_size - 1))

    dead_nodes = list(range(short + long - 1, short + long + dead - 1))
    if dead_nodes:
        template.append((0, dead_nodes[-1]))
        template.extend((dead_nodes[i], dead_nodes[i - 1])
                        for i in range(len(dead_nodes) - 1, 0, -1))

    edges = [(p * stride + u, p * stride + v) for p in range(petals) for u, v in template]
    n_nodes = petals * stride + 1
    depth = _depths(edges, 0)

    # The spine is the chain of hubs: the only nodes every route must pass through.
    hubs = [p * stride for p in range(petals + 1)]
    return Component(edges, n_nodes, 0, depth, [h for h in hubs if h in depth])


def chain(rng: random.Random, *, length: int = 8, **_) -> Component:
    """A bare path. The simplest family that still needs d hops -- useful as a
    sanity check that a method can learn anything at all at depth d."""
    edges = [(i, i + 1) for i in range(length)]
    return Component(edges, length + 1, 0, _depths(edges, 0), list(range(length + 1)))


FAMILIES: dict[str, Callable[..., Component]] = {
    "random_tree": random_tree,
    "star": star,
    "flower": flower,
    "chain": chain,
}


def build(name: str, rng: random.Random, **options) -> Component:
    if name not in FAMILIES:
        raise KeyError(f"unknown graph family {name!r}; have {sorted(FAMILIES)}")
    return FAMILIES[name](rng, **options)
