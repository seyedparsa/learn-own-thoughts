"""Batch collation for continuous-thought training.

Modern rewrite of the upstream ``MyCollator`` (kept verbatim in ``lyot/collator.py``).
Equivalence on the layouts upstream handles is pinned by
``tests/test_collator_equivalence.py``.

Why left padding
----------------
``Coconut.forward`` advances a **single compute window shared by the whole batch**::

    start, end = 0, first_latent_column     # one pair of ints for every sample

So a batch is only well-formed if every sample's latent block starts in the same
column. This collator guarantees that by left-padding shorter prompts until the first
latent tokens line up::

    xxxxxxxxxx<L><L>xxxxx--
    -----xxxxx<L>xxxxxxxx--     <- 5 tokens of left padding
    ---xxxxxxx<L><L>xxxxxxx     <- 3 tokens of left padding
         ^ latents share a column

Latent *counts* may differ per sample (that is what a difficulty-scaled budget
produces); only the start column has to match.

Fixes relative to upstream
--------------------------
* Does not mutate the caller's feature dicts (upstream rewrote them in place, so a
  reused or cached batch was padded twice).
* Pads a singular ``label`` key like ``labels`` (upstream aligned only ``labels``,
  silently emitting a ragged ``label``).
* Requires ``latent_id``; upstream defaulted it to ``None``, which silently skipped
  alignment and produced batches Coconut reads incorrectly.
* Validates the invariants the model depends on (contiguous latents, a position
  before the first latent, a usable ``pad_token_id``) with actionable messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


@dataclass
class Collator:
    """Collate features into a batch whose latent tokens are column-aligned.

    Args:
        tokenizer: supplies ``pad_token_id``; must pad on the right.
        latent_id: id of the ``<|latent|>`` token.
        label_pad_token_id: label fill value ignored by the loss.
    """

    tokenizer: PreTrainedTokenizerBase
    latent_id: int
    label_pad_token_id: int = IGNORE_INDEX

    def __post_init__(self):
        if self.latent_id is None:
            raise ValueError(
                "latent_id is required: without it latent columns cannot be aligned "
                "and Coconut would read hidden states from the wrong positions."
            )
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None; set a pad token "
                '(e.g. STokenizer(...) or tokenizer.pad_token = tokenizer.eos_token).'
            )
        if self.tokenizer.padding_side != "right":
            raise ValueError(
                f"padding_side must be 'right' (got {self.tokenizer.padding_side!r}); "
                "this collator does its own left padding to align latents."
            )

    # -- helpers -------------------------------------------------------------------

    def _first_latent(self, ids: Sequence[int]) -> int:
        """Index of the first latent token, or -1. Also checks contiguity, which
        Coconut requires (one latent is unlocked per forward pass)."""
        positions = [i for i, t in enumerate(ids) if t == self.latent_id]
        if not positions:
            return -1
        if positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError(f"latent tokens must be contiguous, found at {positions}")
        if positions[0] == 0:
            raise ValueError(
                "a latent token cannot be the first token: it takes its embedding "
                "from the hidden state of the preceding position."
            )
        return positions[0]

    @staticmethod
    def _stack(rows: list[list[int]], fill: int, width: int) -> torch.Tensor:
        return torch.tensor(
            [row + [fill] * (width - len(row)) for row in rows], dtype=torch.int64
        )

    # -- entry point ---------------------------------------------------------------

    def __call__(self, features: list[dict[str, Any]], return_tensors=None) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")

        label_key = "label" if "label" in features[0] else "labels"
        has_labels = label_key in features[0]

        # How far each sample must shift right so the latent blocks line up.
        firsts = [self._first_latent(f["input_ids"]) for f in features]
        target = max(firsts)  # -1 when no sample has latents -> no left padding
        shifts = [max(target - first, 0) if first >= 0 else 0 for first in firsts]

        pad_id = self.tokenizer.pad_token_id
        ids_rows, mask_rows, pos_rows, label_rows = [], [], [], []

        for feature, shift in zip(features, shifts):
            ids = list(feature["input_ids"])
            n = len(ids)
            mask = list(feature.get("attention_mask", [1] * n))
            # Real tokens keep positions 0..n-1 regardless of how much padding
            # precedes them, so a prompt encodes identically in any batch.
            pos = list(feature.get("position_ids", range(n)))

            ids_rows.append([pad_id] * shift + ids)
            mask_rows.append([0] * shift + mask)
            pos_rows.append([0] * shift + pos)
            if has_labels:
                label_rows.append([self.label_pad_token_id] * shift + list(feature[label_key]))

        width = max(len(row) for row in ids_rows)
        batch = {
            "input_ids": self._stack(ids_rows, pad_id, width),
            # Right padding is masked out; position id 0 there is inert.
            "attention_mask": self._stack(mask_rows, 0, width),
            "position_ids": self._stack(pos_rows, 0, width),
        }
        if has_labels:
            batch[label_key] = self._stack(label_rows, self.label_pad_token_id, width)

        # Carry any extra per-sample fields through untouched (e.g. `idx`, or ragged
        # metadata such as candidate node sets, which stay a plain list).
        #
        # Keys are taken from the union, not features[0]: a batch can legitimately mix
        # row shapes. At the last curriculum stage some tasks are already finished
        # (answering, no candidate set) while deeper ones are still stepping, and
        # reading one row's keys then indexing every row raised KeyError mid-run.
        # Rows lacking a key get None, which keeps the column ragged rather than
        # silently dropping it.
        reserved = {"input_ids", "attention_mask", "position_ids", "label", "labels"}
        extra_keys = {key for feature in features for key in feature} - reserved
        for key in sorted(extra_keys):
            values = [f.get(key) for f in features]
            batch[key] = (
                torch.tensor(values)
                if all(isinstance(v, (int, float, bool)) for v in values)
                else values
            )
        return batch


# Backwards-compatible alias: upstream call sites construct `MyCollator`.
MyCollator = Collator
