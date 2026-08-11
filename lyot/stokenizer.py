"""Symbolic tokenizer for graph-reasoning tasks.

Modern rewrite of the upstream ``stokenizer.py`` (kept verbatim alongside this file).
Same wire format, but the node-id capacity is configurable instead of hard-coded to 31,
and two upstream bugs are fixed. Equivalence at the default size is pinned by
``tests/test_stokenizer_equivalence.py``.

Wire format::

    <eos> 3 5 | 7 9 [Q] 12 4 [R] 3 <|latent|> <|latent|> [A] 12
          └ edges ┘   └cands┘ └root┘ └── thoughts ──┘    └answer

Every token is one whitespace-delimited symbol, and node ``"k"`` maps to id ``k``, so
data files can store raw node integers. Tokenization is therefore purely structural:
no subword splitting, and no linguistic prior for the model to exploit.
"""

from __future__ import annotations

from typing import Optional, Sequence

from transformers import PreTrainedTokenizer

# Structural markers, in the fixed order they occupy ids [n_nodes, n_nodes + 9).
# Anything optional is appended after these, never inserted among them.
START_LATENT = "<|start-latent|>"
END_LATENT = "<|end-latent|>"
LATENT = "<|latent|>"
EDGE_SEP = "|"
QUESTION = "[Q]"
ROOT = "[R]"
ANSWER = "[A]"
EOS = "<eos>"
NO_ANSWER = "<|no-answer|>"
GRAPH = "[G]"          # optional: introduces the edge list when the question comes first

MARKERS = (
    START_LATENT, END_LATENT, LATENT, EDGE_SEP, QUESTION, ROOT, ANSWER, EOS, NO_ANSWER,
)

DEFAULT_N_NODES = 31  # upstream's capacity: node ids "0".."30", total vocab 40


class STokenizer(PreTrainedTokenizer):
    """Whitespace tokenizer over ``n_nodes`` node ids plus nine structural markers.

    Args:
        n_nodes: how many distinct node ids exist. Ids ``0..n_nodes-1`` are node
            tokens; the markers follow. ``vocab_size == n_nodes + 9 + len(extra_markers)``,
            which must match the model config's ``vocab_size``.
        extra_markers: additional structural symbols a format needs, appended after the
            nine fixed ones so existing ids are unchanged. ``("[G]",)`` for the
            question-first layout.

    Only ``<eos>`` is registered as a special token. The structural markers are
    deliberately *not* special, so ``decode(..., skip_special_tokens=True)`` keeps
    ``[Q] [R] [A]`` — answer parsing splits on ``[A]``.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, n_nodes: int = DEFAULT_N_NODES,
                 extra_markers: Sequence[str] = (), **kwargs):
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")

        self.n_nodes = n_nodes
        # Extras go *after* the nine fixed markers, so every existing id keeps its value and
        # only `vocab_size` grows. A format that needs one declares it (`Method.extra_markers`)
        # and run.py passes it here, so the prompt and the vocabulary cannot disagree -- an
        # unknown marker raises during tokenization rather than becoming a placeholder.
        self.extra_markers = tuple(extra_markers)
        duplicate = set(self.extra_markers) & set(MARKERS)
        if duplicate:
            raise ValueError(f"extra_markers already exist as markers: {sorted(duplicate)}")
        self._vocab: dict[str, int] = {str(i): i for i in range(n_nodes)}
        for offset, marker in enumerate(MARKERS + self.extra_markers):
            self._vocab[marker] = n_nodes + offset
        self._ids_to_tokens: dict[int, str] = {i: t for t, i in self._vocab.items()}

        # The vocabulary must exist before super().__init__(), which validates and
        # registers the special tokens passed here. Upstream assigned pad/eos/bos as
        # attributes *before* calling super(), whose initialiser then reset the
        # special-token registry, leaving pad_token_id / eos_token_id as None.
        kwargs.setdefault("pad_token", EOS)
        kwargs.setdefault("eos_token", EOS)
        kwargs.setdefault("bos_token", EOS)
        super().__init__(n_nodes=n_nodes, extra_markers=self.extra_markers, **kwargs)

    # -- vocabulary ---------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    # Convenience: ids the data pipeline and the Coconut wrapper need.
    @property
    def latent_token_id(self) -> int:
        return self._vocab[LATENT]

    @property
    def start_latent_id(self) -> int:
        return self._vocab[START_LATENT]

    @property
    def end_latent_id(self) -> int:
        return self._vocab[END_LATENT]

    @property
    def answer_token_id(self) -> int:
        return self._vocab[ANSWER]

    def is_node_id(self, token_id: int) -> bool:
        return 0 <= token_id < self.n_nodes

    # -- tokenization -------------------------------------------------------------

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        tokens = text.split()  # str.split() handles newlines and runs of whitespace
        unknown = {t for t in tokens if t not in self._vocab}
        if unknown:
            raise ValueError(
                f"tokens not in vocabulary: {sorted(unknown)}. "
                f"Valid: node ids 0..{self.n_nodes - 1} and {list(MARKERS)}. "
                f"If a graph needs more nodes, construct STokenizer(n_nodes=...) and "
                f"set the model config's vocab_size to n_nodes + {len(MARKERS)}."
            )
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        # No <unk>: an unknown symbol means the data generator emitted an invalid
        # graph, which should fail loudly rather than silently become a placeholder.
        try:
            return self._vocab[token]
        except KeyError:
            raise ValueError(f"token {token!r} not in vocabulary") from None

    def _convert_id_to_token(self, index: int) -> str:
        try:
            return self._ids_to_tokens[index]
        except KeyError:
            raise ValueError(
                f"id {index} out of range for vocab_size {self.vocab_size}"
            ) from None

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return token_ids_0 + eos
        return token_ids_0 + eos + token_ids_1 + eos

    # get_special_tokens_mask is inherited: the base implementation derives the mask
    # from the registered special tokens. Upstream overrode it with a version that
    # indexed self.vocab['<pad>'] / ['<unk>'] -- tokens this vocabulary never had --
    # and raised KeyError whenever already_has_special_tokens=True.

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """No vocab file to write: the vocabulary is fully determined by ``n_nodes``,
        which is round-tripped through tokenizer_config.json."""
        return ()

    def __repr__(self) -> str:
        return f"STokenizer(n_nodes={self.n_nodes}, vocab_size={self.vocab_size})"
