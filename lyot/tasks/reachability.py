"""Graph reachability: which of the candidate nodes is reachable from the root?

Wire format (one whitespace-delimited symbol per token)::

    <eos> 1 2 | 0 3 | 1 4 [Q] 17 15 [R] 1 [A] 17
          └── edges ──┘   └cands┘  └root┘   └answer

Everything structural is derived from ``edges`` on demand and cached per instance:
shortest-path depth, the nodes at each depth on a shortest root->target path, and the
BFS rings. Upstream instead precomputed one view into the data file and shipped a
second file for the other view -- the files agree with what this module computes
(15461/15461 for the golden-path view, 3000/3000 for the ring view), so the extra
files bought nothing but a fork in the pipeline.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional, Sequence

from lyot.tasks.base import Task, register_loader

if TYPE_CHECKING:
    from lyot.tasks.base import TaskSet

EOS = "<eos>"
NO_ANSWER = "<|no-answer|>"
# A content-free token used only to length-match the `[A]` form while leaving the anchor last.
# Boolean rather than a configurable string: `method_options.filler=[D]` on the command line
# parses as the YAML list ['D'], which reached the tokenizer and raised on an unhashable key.
FILLER = "[D]"


class ReachabilityTask(Task):
    """One reachability instance over a directed graph.

    Args:
        edges: directed ``(u, v)`` pairs.
        root: where the search starts.
        target: the reachable candidate -- the correct answer.
        distractor: the unreachable candidate.
        orbit: node -> every image of it across the isomorphic copies, itself included.
            This is what keeps a *promoted* target honestly paired: a method that asks
            about an intermediate node needs the images of *that* node, not of the
            original target. Without it, asking about a mid-arm node against an arm tip
            is separable by out-degree alone -- measured 100% of ATC's non-final stages
            and 0% of its final ones, exactly the set of stages it appeared to solve.

            With two copies the orbit is a pair and this is the old ``twin`` map. With
            more, every offered candidate is depth- and degree-matched, so the guessing
            payoff falls to 1/|orbit| without any candidate becoming easy to eliminate.
        payload: any extra fields from the source file (natural-language ``question``
            and ``answer``, ``idx_to_symbol``), kept so an NL arm can use them later.
    """

    def __init__(self, edges: Sequence[Sequence[int]], root: int, target: int,
                 distractor: int, payload: Optional[dict[str, Any]] = None,
                 orbit: Optional[dict[int, Sequence[int]]] = None,
                 twin: Optional[dict[int, int]] = None):
        self.edges = [(int(u), int(v)) for u, v in edges]
        self.root = int(root)
        self.target = int(target)
        self.distractor = int(distractor)
        # `twin` is the two-copy spelling of `orbit`, accepted so existing callers and
        # pickled tasks keep working.
        if orbit:
            self.orbit = {int(k): tuple(int(x) for x in v) for k, v in orbit.items()}
        elif twin:
            self.orbit = {int(k): (int(k), int(v)) for k, v in twin.items()}
        else:
            self.orbit = {}
        self.payload = payload or {}

    @property
    def twin(self) -> dict[int, int]:
        """The old pair-only view: node -> its first other image."""
        return {n: next(x for x in images if x != n)
                for n, images in self.orbit.items() if len(images) > 1}

    def counterpart(self, node: int) -> int:
        """One unreachable node isomorphic to ``node``.

        Falls back to the task's own distractor when no orbit is known, which is the case
        for tasks loaded from upstream's files.
        """
        images = self.orbit.get(node, ())
        return next((x for x in images if x != node), self.distractor)

    def counterparts(self, node: int) -> tuple[int, ...]:
        """Every image of ``node`` across the copies, itself first.

        The candidate set a question should offer: all of them sit at the same depth with
        the same degrees, so only reachability from the given root separates them.
        """
        images = self.orbit.get(node)
        if not images:
            return (node, self.distractor)
        return (node,) + tuple(x for x in images if x != node)

    def rerooted(self, copy: int) -> "ReachabilityTask":
        """The same graph and the same candidate orbit, asked from a different copy's start.

        This is what makes the root load-bearing. As generated, a graph is only ever seen with
        one root, so "which candidate is reachable" can be answered from the graph alone --
        measured directly on a trapped checkpoint: moving the root between components left the
        prediction distribution unchanged (diagonal fraction 0.236 against a chance 0.250) while
        97.3% of predictions were still one of the offered candidates. The model had learned to
        pick a node identity out of the offered set and to ignore the root entirely.

        Emitting all ``m`` re-rootings of one graph, under one shared node permutation, makes
        that policy unable to fit even the training set: the prompts differ in a single token
        and the labels differ.
        """
        root = self.orbit[self.root][copy]
        target = self.orbit[self.target][copy]
        distractor = next(x for x in self.orbit[target] if x != target)
        return ReachabilityTask(edges=self.edges, root=root, target=target,
                                distractor=distractor, payload=self.payload,
                                orbit=self.orbit)

    # -- structure, all derived ------------------------------------------------------

    @cached_property
    def _adjacency(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for u, v in self.edges:
            out[u].append(v)
        return out

    @cached_property
    def _reverse_adjacency(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for u, v in self.edges:
            out[v].append(u)
        return out

    @cached_property
    def _all_nodes(self) -> list[int]:
        """Every node id appearing in an edge, sorted. Used to draw extra distractors."""
        return sorted({node for edge in self.edges for node in edge})

    @staticmethod
    def _distances(source: int, graph: dict[int, list[int]]) -> dict[int, int]:
        seen = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour in graph[node]:
                if neighbour not in seen:
                    seen[neighbour] = seen[node] + 1
                    queue.append(neighbour)
        return seen

    @cached_property
    def _from_root(self) -> dict[int, int]:
        return self._distances(self.root, self._adjacency)

    @cached_property
    def _to_target(self) -> dict[int, int]:
        return self._distances(self.target, self._reverse_adjacency)

    @property
    def difficulty(self) -> int:
        """Hops on a shortest root->target path."""
        if self.target not in self._from_root:
            raise ValueError(f"target {self.target} is unreachable from root {self.root}")
        return self._from_root[self.target]

    def path_nodes(self, depth: int) -> list[int]:
        """Depth-``depth`` nodes lying on *some* shortest root->target path.

        Upstream's ``neighbor_k`` in the ``*_coconut.json`` files. Usually a single
        node (80% of the time on ProsQA); more when parallel shortest paths exist.
        """
        total = self.difficulty
        return sorted(
            node for node, d in self._from_root.items()
            if d == depth and self._to_target.get(node, total + 1) + d == total
        )

    def ring(self, depth: int) -> list[int]:
        """Every node at exactly ``depth`` hops from the root -- the BFS frontier.

        Upstream's ``neighbor_k`` in the ``*_bfs.json`` files, and much wider than
        ``path_nodes`` (mean 3.4 vs 1.2 on ProsQA), which is why a single-label loss
        over it caps accuracy near 1/|ring|.
        """
        return sorted(node for node, d in self._from_root.items() if d == depth)

    def ball(self, depth: int) -> list[int]:
        """Every node within ``depth`` hops -- the cumulative reachable set."""
        return sorted(node for node, d in self._from_root.items() if d <= depth)

    def shortest_path(self, rng: random.Random) -> list[int]:
        """One root->target path, excluding the root, choosing randomly at each fork."""
        node, path = self.root, []
        for depth in range(1, self.difficulty + 1):
            options = [n for n in self.path_nodes(depth) if n in self._adjacency[node]]
            node = rng.choice(options or self.path_nodes(depth))
            path.append(node)
        return path

    # -- presentation ----------------------------------------------------------------

    def prompt(self, rng: random.Random, reachable: Optional[int] = None,
               start: Optional[int] = None, hide_candidates: bool = False,
               show: str = "both", late: bool = False,
               latents: str = "", n_candidates: Optional[int] = None,
               anchor_mark: str = "[R]", layout: str = "default",
               graph_mark: str = "[G]", filler: bool = False,
               repeat_anchor: bool = False) -> str:
        """Edges and candidates in random order, so neither position nor ordering leaks.

        ``reachable`` overrides which node is offered as the correct candidate; ATC
        uses it to ask about an intermediate node instead of the final target.

        ``start`` moves the ``[R]`` anchor along the route, so the question becomes "from
        here, which candidate is reachable?" with the same two candidates. It is an input,
        never a label, so a method using it still supervises only the answer; walking the
        anchor back toward the root grows how many hops must be inferred while question
        and answer stay fixed.

        An earlier version listed the traversed prefix after the root instead. That failed
        outright -- nothing marked which of the listed nodes was the current position, so
        the model could not tell the anchor from the rest and stayed at chance even where
        one hop with one latent is otherwise solved at 1.000.

        ``show`` controls how many candidates are offered:
            ``both``  the usual pair, shuffled.
            ``one``   a single candidate, drawn at random from the pair -- so half the
                time it is the distractor and the label is a node absent from the prompt.
                Showing the target every time would reduce the task to copying it.
            ``none``  both replaced by a placeholder (the ``hide_candidates`` behaviour).

        ``late`` moves ``[Q]`` and the candidates to *after* the anchor and the latents,
        so the sequence reads ``... [R] root <latents> [Q] cands``. The thoughts are then
        computed before the model has seen which nodes it is being asked about, which is
        the point: with the candidates up front a single latent can hold the answer
        outright (measured: ATC's latent carries it at P=0.996 with zero mass on any
        intermediate), and nothing forces a frontier to be represented. Requires
        ``latents`` to be passed in, since they now sit inside the prompt rather than
        being appended to it.

        ``layout`` reorders the three blocks. ``default`` is ``edges [Q] cands [R] root``;
        ``question-first`` is ``[Q] cands [G] edges [R] root``; ``root-first`` is ``[R] root
        [G] edges [Q] cands``. Only root-first puts the anchor before the edges, which is what
        lets the matching happen while the edges are being read -- see below. This is a
        hypothesis about *when* the matching can happen, not about recency for its own sake.
        With the question last, no edge can be compared against the query while it is being
        read, because the query has not arrived yet; every comparison has to happen at the
        final position, from a single query-dependent attention pattern. With the question
        first, layer 1 can compare each edge against the query as it reads it and mark the
        match in the residual stream, leaving layer 2 to read out the marked position. For a
        two-layer model that is a materially cheaper circuit.

        It is mutually exclusive with ``late`` (which moves the question to the *end*, after
        the latents) and it pulls against ``anchor_mark="|"``: that option wants the root
        adjacent to the answer position, this one wants the edges there. They are competing
        hypotheses about what the query position needs, so they are tested separately.

        ``anchor_mark`` is the token introducing the anchor. The default ``[R]`` is a symbol
        that appears exactly once per sequence, so the model must learn from scratch what
        follows it. Passing ``"|"`` instead makes the tail of the prompt read like an
        *incomplete edge*: every ``| a b`` in the edge list is a tail followed by its head, so
        ``... | root`` is literally the question "what head goes with this tail?", posed in the
        notation the rest of the sequence already uses. If the model has any machinery for
        pairing a tail with its head, this hands it the query in the matching format instead of
        requiring a separate circuit keyed on ``[R]``.

        ``n_candidates`` (only with ``show="both"``) sets how many candidates are offered.
        Left unset it is the whole orbit -- one image per isomorphic copy, so two with the
        twin construction. Above that it pads with arbitrary nodes; below it narrows the
        orbit, keeping the asked node.
        It exists to test *why* offering candidates blocks learning. Measured on
        a depth-10 chain, holding labels and everything else fixed: with candidates hidden the
        stage-0 gate reaches 0.999 within ~28 epochs, while with two offered it sits at 0.50
        indefinitely -- and ``show="one"`` also sits at 0.50, which rules out "the label is
        copyable from the prompt" and leaves the guessing payoff. Emitting *some* candidate
        earns 1/k against 1/|V| for a blind guess, and at k=2 that 0.5 is a resting point the
        optimiser does not leave. If the payoff is the mechanism then raising k should help,
        which is a strange enough prediction to be worth checking. It holds: escape latency
        falls monotonically with the payoff (22, 22, 31, 46 epochs as chance rises through
        0.045, 0.09, 0.14, 0.25), and only k=2 never escapes at all within 800 epochs.

        Which is why the number of copies is the better knob. Padding lowers the payoff but
        draws its extras from every depth, so the discrimination stays two-way; replication
        lowers the payoff *and* makes every candidate depth- and degree-matched. Narrowing
        below the orbit is then the control that separates the two: it moves k at a fixed
        graph, where changing the number of copies necessarily changes the graph's size.

        ``hide_candidates`` is the older spelling of ``show="none"``, kept because runs
        and configs refer to it. It removes the 0.5 chance floor -- the model must
        generate the node from the whole vocabulary -- so a score near chance can no
        longer be confused with having learned to copy one of two given options. Only
        well posed where a single node sits at each depth: true for chains, false for a
        star (800/800 depths are ambiguous with 2 arms).
        """
        edges = list(self.edges)
        rng.shuffle(edges)
        rendered = "|".join(f" {u} {v} " for u, v in edges).strip()
        # A promoted target gets its OWN twin as the distractor, so the two candidates
        # stay isomorphic -- same depth, same in/out degree, one reachable and one not.
        asked = self.target if reachable is None else reachable
        anchor = self.root if start is None else start
        if hide_candidates:
            show = "none"

        if show == "none":
            # Same two slots, so length and positions stay comparable with the normal form.
            offered = f"{NO_ANSWER} {NO_ANSWER}"
        elif show == "one":
            offered = str(rng.choice([asked, self.counterpart(asked)]))
        else:
            # Every image of the asked node, one per isomorphic copy. With two copies this
            # is the familiar pair; with more, k = m comes for free and every candidate is
            # depth- and degree-matched.
            candidates = list(self.counterparts(asked))
            if n_candidates is None:
                pass                                    # the whole orbit, the default
            elif 2 <= n_candidates < len(candidates):
                # Narrow the orbit, keeping the asked node. This is the control for graph
                # size: raising the number of copies necessarily enlarges the graph, so
                # comparing m=4 against m=2 confounds payoff with search breadth. Drawing
                # 2 candidates from a 4-copy graph holds the graph fixed and moves only k.
                others = candidates[1:]
                rng.shuffle(others)
                candidates = [asked] + others[: n_candidates - 1]
            elif n_candidates > len(candidates):
                # Pad past the orbit to lower the guessing payoff further -- see the
                # `n_candidates` note above. These extras are NOT depth-matched (a chain has
                # one node per depth per copy), so they are drawn from the rest of the graph
                # and the isomorphism guarantee covers only the orbit. That is the point:
                # padding is a mechanism probe, replication is the task.
                pool = [n for n in self._all_nodes if n not in candidates]
                rng.shuffle(pool)
                candidates += pool[: n_candidates - len(candidates)]
            rng.shuffle(candidates)
            offered = " ".join(str(c) for c in candidates)

        # Position controls for the `[A]`-marker experiment. `[A]` after the anchor moves the
        # prediction state off the root token, so the root's identity has to be copied forward
        # before it can be used -- plausibly an extra composition step in a two-layer model.
        # Testing that needs the length and position confounds separated:
        #   filler        one token BEFORE the anchor: same total length as the `[A]` form,
        #                 but the final position is still the anchor.
        #   repeat_anchor the anchor twice: the prediction state moves, yet the root is still
        #                 the immediately preceding token.
        # If depth selectively repairs only the `[A]` form, the depth account survives; if the
        # filler or repeated form is equally impaired, it does not.
        anchor_text = f"{anchor} {anchor}" if repeat_anchor else str(anchor)
        lead = f"{FILLER} " if filler else ""
        if layout == "root-first":
            # `[R] root [G] edges [Q] cands`. This is the ordering the two-layer depth
            # argument actually calls for. At an edge-head token b, causal attention can
            # reach its own tail a AND the root r, both of which precede it, so layer 1 can
            # test a == r and mark the position; layer 2 only has to read the marked entry.
            # With the anchor last, no edge can be tested against the root while it is read,
            # so every comparison has to happen at the final position from one
            # query-dependent attention pattern.
            #
            # With nothing offered the `[Q]` block is dropped entirely, so the edges sit
            # against the answer position: `[R] root [G] edges`.
            tail = "" if show == "none" else f" [Q] {offered}"
            return (f"{EOS} {lead}{anchor_mark} {anchor_text} "
                    f"{graph_mark} {rendered}{tail}")
        if layout == "question-first":
            # Candidates first, anchor last. Weaker than root-first: an edge-head token can
            # test whether its node is *in* the candidate list, but in the matched
            # construction all k candidates are analogous heads in the k components, so this
            # marks k positions and still cannot choose among them until the root arrives.
            return (f"{EOS} [Q] {offered} {graph_mark} {rendered} "
                    f"{lead}{anchor_mark} {anchor_text}")
        if late:
            return (f"{EOS} {rendered} {lead}{anchor_mark} {anchor_text}"
                    f"{latents}[Q] {offered}")
        return f"{EOS} {rendered} [Q] {offered} {lead}{anchor_mark} {anchor_text}"

    @property
    def answer(self) -> str:
        return str(self.target)

    def relabelling(self, rng: random.Random) -> dict[int, int]:
        """A random node-id permutation for this graph, so several tasks can share one."""
        nodes = sorted({n for edge in self.edges for n in edge})
        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        return dict(zip(nodes, shuffled))

    def perturbed(self, rng: random.Random,
                  mapping: Optional[dict[int, int]] = None) -> "ReachabilityTask":
        """Permute node ids, keeping the graph isomorphic (eos's ``shuffle_nodes``).

        Pass ``mapping`` to relabel several tasks identically. The re-rooted variants of one
        graph must share it: their whole point is that the prompts are identical but for the
        root token, which a per-task permutation would destroy.
        """
        mapping = self.relabelling(rng) if mapping is None else mapping
        return ReachabilityTask(
            edges=[(mapping[u], mapping[v]) for u, v in self.edges],
            root=mapping.get(self.root, self.root),
            target=mapping.get(self.target, self.target),
            distractor=mapping.get(self.distractor, self.distractor),
            payload=self.payload,
            orbit={mapping.get(n, n): tuple(mapping.get(x, x) for x in images)
                   for n, images in self.orbit.items()},
        )

    def __repr__(self) -> str:
        return (f"ReachabilityTask(root={self.root}, target={self.target}, "
                f"distractor={self.distractor}, hops={self.difficulty}, "
                f"|E|={len(self.edges)})")


def from_records(records: list[dict[str, Any]]) -> list[ReachabilityTask]:
    """Build tasks from upstream's JSON records.

    Any ``neighbor_k`` present is ignored: it is derivable, the two upstream variants
    disagree about what it means, and recomputing keeps one file usable by every method.
    """
    reserved = {"edges", "root", "target", "neg_target", "neighbor_k", "neg_neighbor_k", "steps"}
    return [
        ReachabilityTask(
            edges=record["edges"],
            root=record["root"],
            target=record["target"],
            distractor=record["neg_target"],
            payload={k: v for k, v in record.items() if k not in reserved},
        )
        for record in records
    ]


register_loader("reachability", from_records)


# -- generation ------------------------------------------------------------------------

def _replicate(component, target: int, n_components: int = 2) -> "ReachabilityTask":
    """One task: the component plus ``n_components - 1`` unreachable isomorphic copies.

    Node ids are left as the offsets produce them; the caller permutes them. Splitting it
    that way means the permutation lives in exactly one place (``perturbed``).

    Why more than two copies is worth having. Offering the target against *k* distractors
    lowers the payoff for the lazy policy "emit some offered node" from 1/2 to 1/k, and a
    lower payoff measurably shortens the time to escape it. But padding the candidate list
    with arbitrary nodes -- the earlier way of raising k -- draws them from every depth, so a
    model that has learned depth discards them for free. The payoff falls while the actual
    discrimination stays two-way, which is a weaker manipulation than it looks.

    Replicating the component instead makes every candidate the target's image under an
    isomorphism: same depth, same in-degree, same out-degree, differing only in which copy
    holds the root. Depth cannot eliminate any of them, so the root has to be used. And
    exactly one image is reachable, so exact match still tops out at 1.0 -- unlike a ring
    curriculum, where several labels are valid and the ceiling is 1/|ring|.
    """
    if n_components < 2:
        raise ValueError(f"need at least two components to pose a choice, got {n_components}")
    size = component.n_nodes
    edges = []
    for copy in range(n_components):
        shift = copy * size
        edges += [(u + shift, v + shift) for u, v in component.edges]
    # node -> every image of it across the copies, itself included, in copy order.
    orbit = {}
    for base in range(size):
        images = tuple(base + copy * size for copy in range(n_components))
        for node in images:
            orbit[node] = images
    return ReachabilityTask(edges=edges, root=component.start, target=target,
                            distractor=target + size, orbit=orbit)


def generate(
    family: str,
    *,
    n_graphs: int,
    seed: int = 0,
    name: Optional[str] = None,
    targets: str = "deepest",
    depths: Optional[Sequence[int]] = None,
    n_components: int = 2,
    roots: str = "one",
    **options,
) -> "TaskSet":
    """Build a split of reachability tasks from a generated graph family.

    By default one graph yields one task, targeting the far end of its spine, so a task
    *is* the full problem and easier versions of it come from the curriculum rather
    than the data. Pass ``targets="all"`` for the data-curriculum setting, where one
    graph also yields the shallower instances as separate tasks.

    Node ids are permuted per task. Without that the copy's ``+n_nodes`` offset *is*
    the answer -- the target is always the smaller candidate (measured 30/30), and a
    2-layer model reached 1.000 at every depth within two epochs by reading it off.
    This is not the same as the optional ``shuffle_nodes`` augmentation: that is a
    training-time flag, off by default and not applied to the eval split, so it cannot
    close a leak in the data itself.

    Args:
        family: a key of ``lyot.tasks.graphs.FAMILIES``.
        n_graphs: how many base structures to draw; the task count is a multiple of it.
        targets: ``deepest`` (one full-difficulty task per graph) or ``all``
            (additionally the shallower instances -- the data curriculum).
        depths: restrict to these difficulties.
        n_components: how many isomorphic copies of the drawn structure to place in each
            graph. One holds the root, the rest are unreachable. 2 is the twin setting;
            higher values offer more depth-matched candidates, which cuts the payoff for
            guessing among them to 1/n without making any of them eliminable by depth.
        roots: ``one`` puts the root in the first copy, so a graph is only ever seen with a
            single root. ``all`` emits one task per copy instead -- same edges, same candidate
            orbit, one shared node permutation, only the root token and the label differ.
            That turns each graph into a set of counterfactuals and makes a root-ignoring
            policy unable to fit the training set; without it, a trapped model was measured
            to ignore the root completely (see ``rerooted``). Multiplies the task count by
            ``n_components``.
        **options: passed to the family (``n``, ``arms``, ``length``, ``petals``, ...).
    """
    from lyot.tasks.base import TaskSet
    from lyot.tasks.graphs import build as build_graph

    rng = random.Random(seed)
    tasks: list[ReachabilityTask] = []
    for graph_index in range(n_graphs):
        component = build_graph(family, rng, **options)
        for depth, target in component.targets(targets):
            if depths is not None and depth not in depths:
                continue
            base = _replicate(component, target, n_components)
            # Permute per task, not per graph: one permutation shared across a graph's
            # *depths* would let ids correlate between the tasks in a batch. Sharing one
            # across a graph's *re-rootings* is the opposite -- it is what makes them
            # counterfactuals of each other -- so `roots="all"` shares deliberately.
            if roots == "all":
                mapping = base.relabelling(rng)
                variants = [base.rerooted(copy).perturbed(rng, mapping)
                            for copy in range(n_components)]
            elif roots == "one":
                variants = [base.perturbed(rng)]
            else:
                raise ValueError(f"roots must be 'one' or 'all', got {roots!r}")
            for task in variants:
                task.payload = {"family": family, "max_depth": component.max_depth,
                                # Which drawn graph this came from. The re-rootings of one
                                # graph must be relabelled *together* or they stop being
                                # counterfactuals of each other, and `generate` shuffles the
                                # task list, so the grouping cannot be recovered from order.
                                # The seed is part of the key because a mixture (`length:
                                # [1,2,3]`) calls generate() once per value, each restarting
                                # the index at 0 -- sharing a bare index across those applied
                                # a small graph's permutation to a larger graph's node set.
                                "graph": f"{seed}:{graph_index}"}
                tasks.append(task)
    rng.shuffle(tasks)
    return TaskSet(name=name or f"{family}-{n_graphs}", tasks=tasks)




