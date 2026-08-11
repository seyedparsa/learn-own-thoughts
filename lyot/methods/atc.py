"""ATC: raise question difficulty and thinking budget together.

Stage ``s`` asks about a node at depth ``s+1`` -- an intermediate node of the full
problem, promoted to be the answer and offered as one of the two candidates -- with
``s`` latents and ``[A]`` as the only label. No step is ever supervised, which is the
point: the latents are unsupervised, and only the answer is taught.

With ``n`` thoughts you answer the ``n+1``-hop question, so stage 0 is a one-hop
lookup with no thinking rather than the full problem with none, and the deepest stage
recovers the original task. The distractor stays the task's own.

Which nodes are eligible at each depth is the ``curriculum`` axis: ``path`` (the
default, and how ATC has been run) promotes a node on a shortest path, so the question
stays answerable; ``ring`` promotes any node at that distance, making the curriculum
about frontiers rather than solutions. ``ball`` is a poor fit here -- it contains the
root, and ATC *asks whether* the promoted node is reachable, so "is the root reachable
from the root?" is true without reading an edge.

Known weakness, measured rather than suspected: on ProsQA the distractor is
unreachable from the root in 257/257 instances, so "which of these two is reachable?"
remains separable by local features at every stage. The trained model's latents are
inert -- accuracy is flat from 0 to 4 latents, and its 0-latent score (0.720) matches
the no-CoT baseline (0.720). See scripts/ablate_latents.py. Fixing that means pairing
the label against a harder distractor, which is a deliberate next experiment, not
something this class should decide on its own.
"""

from __future__ import annotations

import random
from typing import Sequence

from lyot.methods.base import NO_ANSWER, Example, Method, register_method
from lyot.tasks import Task


@register_method
class ATC(Method):
    """Answer-only supervision over a difficulty curriculum.

    Args:
        no_answer_prob: probability of asking with fewer latents than the depth needs,
            labelled ``<|no-answer|>`` -- teaches abstention rather than guessing.
        answer_mark: append ``[A]`` after the thoughts. Default True, which is what every ATC
            run so far has done and is self-consistent -- each ATC stage poses a legitimate
            easier question, and its label answers that question. Set False to ablate it: with
            the candidates hidden, toggling this alone moved a depth-10 chain from 0.057 to
            1.000, plausibly because the label then sits adjacent to the last latent rather
            than a stage-invariant token away from it. See ``Method.close``.
        show: how many candidates the question offers -- ``both``, ``one`` (drawn at random,
            so half the time the shown node is the distractor and the label is absent from
            the prompt), or ``none`` (placeholders). Not cosmetic either: with two candidates
            shown, the label is always one of them, so "emit a candidate" buys 1/2 against
            1/|vocab| for guessing. That is a cheap local optimum, and escaping it requires
            learning reachability against a gradient that already explains half the labels.
            ``both`` on a promoted question failed in the bridge factorial whether or not the
            marker was present (0.540 / 0.277), gate pinned near 0.5, while hiding the
            candidates on the same question reached 1.000 -- so this is a second, independent
            failure mode rather than another face of the marker.
    """

    name = "atc"

    def budget_for(self, stage: int, depth: int, task: Task) -> int:
        """Latents granted at this stage. ATC's own fencepost: ``stage`` thoughts answer the
        depth-``stage+1`` question, so this is one short of the hop count by design.

        The single hook subclasses override. It used to be that ``atc-matched`` reimplemented
        the whole of ``example()`` to change this one number, and the duplicate silently
        missed both the ``candidates=`` and ``n_candidates`` fixes -- which pinned every
        ``atc-matched`` star run at stage 0. Overriding one integer makes that impossible.

        ``budget="depth-1"`` caps the grant per instance at ``depth-1`` -- the exact fencepost
        for a graph of *this* difficulty, rather than a single ``max_latent_stage`` shared
        across a mixed-difficulty family. ``depth`` here is ``min(stage+1, difficulty)``, so
        this returns ``min(stage, difficulty-1)``: a tree never gets more thoughts than its own
        depth needs. It is a no-op on fixed-difficulty tasks whose ``max_latent_stage`` already
        equals ``difficulty-1`` (the chain/star runs), and removes the +1 over-provisioning on
        the shallow members of a tree family. The default ``"stage"`` is plain ATC.

        ``budget="max"`` decouples the two curricula: the grant is what the instance would get
        at the *final* stage -- ``min(difficulty-1, max_latent_stage)`` -- from the first epoch,
        while the question's depth still follows the stage. It is the control for whether the
        budget has to grow *with* difficulty or only has to be sufficient: ATC raises both
        together, and this raises difficulty alone. Evaluation is identical to ``depth-1``, so
        the arms differ only in what training saw.
        """
        setting = self.options.get("budget", "stage")
        if setting == "depth-1":
            return max(depth - 1, 0)
        if setting == "max":
            return min(max(task.difficulty - 1, 0), self.max_latent_stage)
        return stage

    def build_prompt(self, task: Task, rng: random.Random, label: int, show: str,
                     latents: str) -> tuple[str, str]:
        """Return ``(prompt, trailing_latents)`` -- the question, and whatever ``close()``
        must still append after it.

        The one hook a presentation variant overrides. Everything else about an ATC row --
        which label is drawn, the acceptable set behind ``set_acc``, ``n_candidates``,
        ``answer_mark``, the abstention branch -- is decided once in ``example()``. Two bugs
        came from variants reimplementing ``example()`` to change only the prompt or only the
        budget, and silently missing later fixes to the parent.
        """
        prompt = task.prompt(rng, reachable=label, show=show,
                             n_candidates=self.options.get("n_candidates"),
                             anchor_mark=self.options.get("anchor_mark", "[R]"),
                             layout=self.options.get("layout", "default"),
                             filler=self.options.get("filler", ""),
                             repeat_anchor=self.options.get("repeat_anchor", False),
                             graph_mark=self.options.get("graph_mark", "[G]"),
                             hide_candidates=self.options.get("hide_candidates", False))
        return prompt, latents

    def build_eval_prompt(self, task: Task, rng: random.Random, show: str,
                          latents: str) -> tuple[str, str]:
        """The benchmark's question, in the same layout training used."""
        prompt = task.prompt(rng, show=show,
                             n_candidates=self.options.get("n_candidates"),
                             anchor_mark=self.options.get("anchor_mark", "[R]"),
                             layout=self.options.get("layout", "default"),
                             filler=self.options.get("filler", ""),
                             repeat_anchor=self.options.get("repeat_anchor", False),
                             graph_mark=self.options.get("graph_mark", "[G]"))
        return prompt, latents

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        depth = min(stage + 1, task.difficulty)
        options = self.intermediates(task, depth)
        label = rng.choice(options)
        show = self.options.get("show", "both")
        # `hidden_prob`: hide the candidates on this fraction of TRAINING examples only.
        # A model trained entirely without candidates solves stage 0 at 0.970 and still
        # scores 0.913 when candidates are inserted, a format it never saw -- so the
        # candidate tokens do not break execution, they block the circuit from forming. A
        # small hidden fraction is then a way to supply the missing initial alignment while
        # the task itself stays candidate-visible. Evaluation is untouched, so the benchmark
        # remains the visible task.
        hidden_prob = self.options.get("hidden_prob", 0.0)
        if hidden_prob and show != "none" and rng.random() < hidden_prob:
            show = "none"
        latents = self.latents(self.budget_for(stage, depth, task))
        prompt, tail = self.build_prompt(task, rng, label, show, latents)

        no_answer_prob = self.options.get("no_answer_prob", 0.0)
        if no_answer_prob and stage >= 1 and rng.random() < no_answer_prob:
            short = rng.randint(0, stage - 1)
            return Example(f"{prompt}{self.latents(short)}[A]", NO_ANSWER, depth)
        # is_final stays True: every ATC row answers the (easier) question it posed. The
        # marker is a separate, explicit choice -- see Method.close.
        mark = self.options.get("answer_mark", True)
        return Example(self.close(prompt, tail, mark), str(label), depth,
                       candidates=self._acceptable(task, depth, show))

    def eval_example(self, task: Task, rng: random.Random) -> Example:
        show = self.options.get("show", "both")
        if self.options.get("hide_candidates", False):
            show = "none"
        # The benchmark budget must match the training fencepost. With `budget="depth-1"`
        # each instance is scored at its own depth-1 thoughts, not the shared difficulty.
        if self.options.get("budget", "stage") in ("depth-1", "max"):
            budget = min(max(task.difficulty - 1, 0), self.max_latent_stage)
        else:
            budget = min(task.difficulty, self.max_latent_stage)
        prompt, tail = self.build_eval_prompt(task, rng, show, self.latents(budget))
        mark = self.options.get("answer_mark", True)
        # Carry the frontier when the prompt names nothing (show=none), so the benchmark logs
        # set_acc -- "did it name a valid deepest node" -- alongside exact match, which is
        # capped near 1/|ring| on a branching graph. On a chain the ring is one node, so the
        # set is empty and this is exact match, unchanged.
        return Example(self.close(prompt, tail, mark), task.answer, task.difficulty,
                       candidates=self._acceptable(task, task.difficulty, show))

    @staticmethod
    def _acceptable(task: Task, depth: int, show: str) -> tuple[int, ...]:
        """Which answers the question admits -- what ``set_acc`` and the set loss score against.

        Only set-valued when the prompt does not say which node is meant. With ``show="none"``
        the question is exactly "name a node at this depth", so **every node at that depth
        answers it**, whatever the label happened to be. That set is the BFS frontier
        ``task.ring(depth)``, not the curriculum's label set: with ``curriculum="path"`` the
        label is the one node en route to a target the hidden prompt never names, so exact
        match against it is unachievable on any branching graph -- measured 1000/1000 depths
        ambiguous on a two-arm star, against 0/1000 on a chain.

        Separating the two makes the ambiguity survivable rather than fatal. Cross-entropy
        still cannot fall below ln|ring| -- the model has no way to know which member was
        drawn -- but accuracy reaches 1.0 as soon as it can name *a* correct node, so a 0.9
        gate is attainable and the curriculum can move. On a chain the ring is a single node
        and this is identical to exact match, so chain results are unaffected.

        Getting this wrong is invisible: an empty set means no ``set_acc`` is logged, the
        staging gate silently falls back to exact match, and a 0.9 threshold against a 0.5
        ceiling pins the curriculum at stage 0 forever. That is what happened to the first
        ATC-on-star runs.
        """
        if show != "none":
            return ()                       # the prompt names the node; one answer is right
        frontier = task.ring(depth) if hasattr(task, "ring") else ()
        return tuple(frontier) if len(frontier) > 1 else ()


@register_method
class BudgetATC(Method):
    """Answer-only supervision, but the *budget* is staged instead of the question.

    ATC stages difficulty: stage ``s`` asks an easier, depth-``s+1`` question. That fails
    on the twin star -- at chance where Coconut reaches 0.99 -- and no learning rate,
    depth, or epoch budget moves it. The suspicion is that each of its stages is a
    *different, self-contained* problem, so nothing forces one stage's solution to become
    the next one's first step.

    Here the question is always the real one, at full difficulty, and what grows is how
    many latents the model is allowed. Stage ``s`` gives ``s`` thoughts for a problem that
    needs ``difficulty`` of them, so early stages are genuinely under-budgeted -- the
    model must do as well as it can and then learn to use each new thought as it arrives.
    Still no step labels: the only supervision is the final answer.
    """

    name = "atc-budget"

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        budget = min(stage, task.difficulty)
        return Example(f"{task.prompt(rng)}{self.latents(budget)}[A]",
                       task.answer, task.difficulty)


@register_method
class AdvancedStartATC(Method):
    """Answer-only supervision, with the starting point walked back toward the root.

    The measured obstacle for ATC is credit assignment across latents, not capability: a
    one-hop lookup is unlearnable with 0 latents (no-cot scores 0.480 at depth 1) and solved
    with 1 (atc reaches 1.000), but two latents never learn to compose. Nothing in an
    answer-only loss says what the first thought should hold, so there is no gradient path
    to a two-step solution.

    The target, the distractor and the graph never change. What changes is where the search
    starts: at stage 0 the anchor sits one hop from the target, and each stage walks it back
    one node while granting one more latent. So stage 0 is exactly the configuration we know
    is solvable -- one hop, one latent, anchored at ``[R]`` -- and every later stage asks for
    the same shape of hop one position earlier, which is the incremental signal plain ATC has
    no way to obtain.

    The anchor is an input, never a label. No intermediate node is ever supervised.

    This replaces an earlier design that listed the traversed prefix after the root. That
    version stayed at chance even at stage 0, because nothing marked which listed node was
    the current position -- the model could not locate the anchor.
    """

    name = "atc-advance"

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        hops = min(stage + 1, task.difficulty)          # hops left to infer
        depth_of_anchor = task.difficulty - hops
        anchor = (task.root if depth_of_anchor == 0
                  else task.path_nodes(depth_of_anchor)[0])
        # One latent per hop: a hop needs a thought to happen in. Zero latents cannot do
        # even one hop here (no-cot 0.480 at depth 1) while one latent solves it (1.000).
        budget = min(hops, self.max_latent_stage)
        prompt = task.prompt(rng, start=anchor)
        return Example(f"{prompt}{self.latents(budget)}[A]", task.answer, hops)


@register_method
class MatchedATC(ATC):
    """ATC with one latent per hop the question actually requires.

    Plain ATC follows the specified fencepost -- with ``n`` thoughts you answer the
    ``n+1``-hop question -- so stage ``s`` grants ``s`` latents for a depth-``s+1`` question.
    Measured, that is short by exactly one at every stage: zero latents cannot do even one hop
    here (no-cot scores 0.480 at depth 1) while one latent solves it outright (1.000). Stage 0
    is therefore unlearnable by construction, and every later stage inherits the same deficit.

    ATC appeared to solve its early stages only because the distractor was held fixed, which
    made them separable by out-degree alone -- 100% of the stages it "solved" and 0% of those
    it failed. With that pairing corrected the deficit is exposed and ATC sits at chance
    everywhere.

    This variant keeps everything else identical and grants ``depth`` latents for a
    depth-``depth`` question. It is still answer-only: no intermediate is ever supervised.
    """

    name = "atc-matched"

    def budget_for(self, stage: int, depth: int, task: Task) -> int:
        return min(depth, self.max_latent_stage)   # one thought per hop, not depth - 1


@register_method
class LateQuestionATC(ATC):
    """ATC with the question moved *after* the thoughts.

    Every ATC arm so far shows the candidates before the latents run, and the logit lens
    says exactly what the model does with that: its single latent holds **the answer** at
    P=0.996, with zero mass on any intermediate, in 300/300 cases. It computes the answer
    in the forward pass and parks it in the thought. Nothing is left for a second latent to
    refine, which is the mechanism behind "answer-only supervision learns one hop and does
    not learn two".

    Here the prompt reads ``... [R] root <latents> [Q] candidates [A]``. The thoughts are
    computed before the model knows which nodes it will be asked about, so a thought cannot
    be the answer -- the best it can encode is what is reachable from the root so far, which
    is a frontier. That is the representation Coconut's step supervision produces and that
    answer-only supervision has never been shown to discover on its own.

    ``show`` selects how many candidates are revealed at the end:
        ``both``  the usual pair -- "which of these two?"
        ``one``   a single candidate, drawn at random, and the label is still the true
            target. Half the time the shown node is the distractor, so copying fails.

    Everything else matches ATC as specified: stage ``s`` asks a depth-``s+1`` question
    about a promoted intermediate with ``s`` latents -- the same fencepost Coconut uses,
    where stage ``s`` also grants ``s`` latents and predicts the depth-``s+1`` node -- and
    only the answer is supervised.

    ``answer_mark`` (default ``False``) appends ``[A]`` to intermediate rows. Coconut omits
    it there, reserving the marker for rows that actually answer, and the bridge factorial
    found the choice decisive rather than cosmetic: every arm with ``[A]`` on intermediate
    rows failed to clear stage 0 in 780+ epochs, while three of four without it solved a
    depth-10 chain. The deepest stage answers the real question and always carries it.
    """

    name = "atc-late"

    # The latents sit *inside* the prompt here, before [Q], so close() appends none.
    def build_prompt(self, task: Task, rng: random.Random, label: int, show: str,
                     latents: str) -> tuple[str, str]:
        return task.prompt(rng, reachable=label, late=True, show=show,
                           n_candidates=self.options.get("n_candidates"),
                             anchor_mark=self.options.get("anchor_mark", "[R]"),
                             layout=self.options.get("layout", "default"),
                             filler=self.options.get("filler", ""),
                             repeat_anchor=self.options.get("repeat_anchor", False),
                             graph_mark=self.options.get("graph_mark", "[G]"),
                           latents=latents), ""

    def build_eval_prompt(self, task: Task, rng: random.Random, show: str,
                          latents: str) -> tuple[str, str]:
        return task.prompt(rng, late=True, show=show,
                           n_candidates=self.options.get("n_candidates"),
                             anchor_mark=self.options.get("anchor_mark", "[R]"),
                             layout=self.options.get("layout", "default"),
                             filler=self.options.get("filler", ""),
                             repeat_anchor=self.options.get("repeat_anchor", False),
                             graph_mark=self.options.get("graph_mark", "[G]"),
                           latents=latents), ""


@register_method
class LateMatchedATC(LateQuestionATC):
    """The late question, with one latent per hop instead of ATC's fencepost.

    This exists because every earlier ``atc-late`` run was untestable. ATC's fencepost gives
    stage ``s`` exactly ``s`` latents, so stage 0 has *zero* -- and all twelve of those runs
    died at stage 0, across roughly 3,900 epochs. So the prompt never contained a thought, and
    the hypothesis the method exists to test (that latents forced to precede the question hold
    a frontier rather than the answer) was never exercised. A probe of one checkpoint appeared
    to show the thoughts encoding depth but not component; that reading was wrong, because at
    stage 0 there are no thoughts to encode anything.

    Granting ``depth`` latents makes stage 0 a one-hop question with one thought, which is the
    configuration measured to be learnable elsewhere (no-cot scores 0.480 at depth 1 with none;
    one latent reaches 1.000). If the curriculum can then advance, the superposition question
    becomes answerable for the first time.
    """

    name = "atc-late-matched"

    def budget_for(self, stage: int, depth: int, task: Task) -> int:
        return min(depth, self.max_latent_stage)


@register_method
class VerifyATC(Method):
    """Late question, one candidate, and a yes/no label: "is this node reachable?"

    The most constrained arm. The prompt is ``... [R] root <latents> [Q] node [A]`` and the
    label is ``1`` when the offered node is the one reachable at this depth and ``0`` when it
    is the distractor -- so the thoughts run before the question is known *and* the answer
    carries no information about which node it was.

    The reasoning: in every generative arm the label is a node, so a latent that memorises
    the answer is rewarded. Here it cannot be -- a single bit cannot name a node, and the
    node is not known when the thoughts are computed. To be right more than half the time the
    model has to hold what is reachable and test membership against it. If a frontier is ever
    going to be forced out of answer-only supervision, this is the shape that forces it.

    Both labels appear equally often by construction, so chance is 0.500.

    Caveat, deliberate and worth stating: ``0`` and ``1`` are also node ids, since the
    vocabulary is ``str(i)`` for each node. The label token therefore shares an embedding with
    nodes 0 and 1. Position disambiguates it -- only ever these two tokens follow ``[A]`` in
    this arm -- but the collision is real, and a dedicated pair of yes/no markers would be
    cleaner if this variant proves worth pursuing.

    NOT comparable to the other arms on ``eval/acc``: this benchmark is a 2-way decision with
    chance 0.500, where theirs is generating a node from the whole vocabulary.
    """

    name = "atc-verify"

    @staticmethod
    def _pose(task: Task, rng: random.Random, asked: int, latents: str) -> tuple[str, str]:
        """One prompt offering a single node, and the bit saying whether it is reachable."""
        truthful = rng.random() < 0.5
        offered = asked if truthful else task.counterpart(asked)
        edges = list(task.edges)
        rng.shuffle(edges)
        rendered = "|".join(f" {u} {v} " for u, v in edges).strip()
        prompt = f"<eos> {rendered} [R] {task.root}{latents}[Q] {offered} [A]"
        return prompt, ("1" if truthful else "0")

    def example(self, task: Task, stage: int, rng: random.Random) -> Example:
        depth = min(stage + 1, task.difficulty)
        label = rng.choice(self.intermediates(task, depth))
        prompt, bit = self._pose(task, rng, label, self.latents(stage))
        return Example(prompt, bit, depth)

    def eval_example(self, task: Task, rng: random.Random) -> Example:
        budget = min(task.difficulty, self.max_latent_stage)
        prompt, bit = self._pose(task, rng, task.target, self.latents(budget))
        return Example(prompt, bit, task.difficulty)
