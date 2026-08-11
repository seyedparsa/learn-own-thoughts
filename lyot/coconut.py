# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modernized reimplementation of the Coconut wrapper (see lyot/coconut.py for the
# verbatim upstream original). Behaviourally equivalent, but:
#   * version-agnostic KV-cache handling (works on transformers 4.x and 5.x)
#   * vectorized continuous-thought write-back instead of a per-token Python rebuild
#   * a single forward call per pass instead of duplicated first/cached branches
#   * explicit validation of the latent layout the algorithm requires
# Equivalence against the original is pinned by tests/test_coconut_equivalence.py.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_N_LATENT = 8


@dataclass
class Outputs:
    """Superset of the upstream namedtuple (same attribute names, plus the cache)."""

    loss: Optional[torch.Tensor]
    inputs_embeds: torch.Tensor
    logits: torch.Tensor
    past_key_values: Any = None


class Coconut(nn.Module):
    """Chain of Continuous Thought.

    Wraps a causal LM. Positions holding ``latent_token_id`` do not use that token's
    embedding: each is replaced by the model's own last hidden state from the preceding
    position. Because thought *k+1* depends on the output that produced thought *k*, the
    sequence is built over ``n_latent + 1`` forward passes, each unlocking one more
    position and reusing the KV cache for everything already computed:

        pass 0: [....prompt....]                      -> h  writes thought 1
        pass 1:                 [t1]                  -> h  writes thought 2
        pass 2:                     [t2]              -> h  writes thought 3
        pass 3:                         [...rest...]  -> logits for the answer

    Latent positions carry ``-100`` labels, so the thoughts receive no direct
    supervision; they are trained purely by gradients flowing from the answer loss.

    Layout requirement: within a batch the latent positions must be **contiguous** and
    **column-aligned** across samples (the compute window is shared). The collator is
    responsible for left-padding to achieve this. Counts may differ per sample.

    ``set_loss`` swaps the intermediate objective for eos's BFS loss -- see
    ``_set_loss``. It changes nothing about the latents themselves.
    """

    def __init__(
        self,
        base_causallm: nn.Module,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        max_n_latent: int = MAX_N_LATENT,
        loss_kind: str = "ce",
        set_loss: bool = False,   # deprecated alias for loss_kind="log-of-sum"
        backprop_last: int | None = None,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.max_n_latent = max_n_latent
        # Aggregation of the loss over a set-valued label (the frontier/ring). Three kinds:
        #   ce           ordinary cross-entropy on the single sampled member.
        #   log-of-sum   -log SUM_{v in set} p(v): mass anywhere on the set is optimal, so it
        #                credits naming ANY member -- but permits collapsing onto one.
        #   sum-of-log   -(1/k) SUM_{v in set} log p(v) (cross-entropy to uniform-over-set):
        #                optimal only when mass is spread across ALL members, so it forces the
        #                latent to carry the whole frontier -- superposition -- rather than
        #                committing early. `set` is the old name for log-of-sum.
        if set_loss and loss_kind == "ce":
            loss_kind = "log-of-sum"
        loss_kind = {"set": "log-of-sum"}.get(loss_kind, loss_kind)
        if loss_kind not in ("ce", "log-of-sum", "sum-of-log"):
            raise ValueError(f"unknown loss_kind {loss_kind!r}; "
                             "have ce, log-of-sum (=set), sum-of-log")
        self.loss_kind = loss_kind
        # None = backpropagate through every pass (the original behaviour).
        self.backprop_last = backprop_last
        # Works for every HF causal LM (for GPT-2 this is transformer.wte, i.e. the
        # same tensor upstream reaches for via its isinstance special case).
        self.embedding = base_causallm.get_input_embeddings()
        self.gen_forward_cnt = 0

    # -- internals ---------------------------------------------------------------

    def _latent_layout(self, input_ids: torch.Tensor) -> tuple[int, torch.Tensor, int]:
        """Locate the continuous thoughts and validate the layout.

        Returns ``(first_col, counts, n_passes)``; ``first_col`` is -1 when the batch
        holds no latents at all.
        """
        is_latent = input_ids == self.latent_token_id
        counts = is_latent.sum(dim=1)
        n_passes = int(counts.max())
        if n_passes == 0:
            return -1, counts, 0

        has = counts > 0
        first = torch.argmax(is_latent.int(), dim=1)
        last = input_ids.shape[1] - 1 - torch.argmax(is_latent.int().flip(1), dim=1)

        first_col = int(first[has].min())
        if not bool((first[has] == first_col).all()):
            raise ValueError(
                "Coconut requires the first latent token to sit in the same column for "
                f"every sample (the compute window is shared); got columns {first[has].tolist()}. "
                "Left-pad the batch (see MyCollator) to align them."
            )
        if not bool(((last[has] - first[has] + 1) == counts[has]).all()):
            raise ValueError("Coconut requires each sample's latent tokens to be contiguous.")
        if first_col == 0:
            raise ValueError("A latent token needs a preceding position to read its hidden state from.")
        return first_col, counts, n_passes

    # -- losses --------------------------------------------------------------------

    def _set_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                  candidate_nodes: list[Optional[list[int]]]) -> torch.Tensor:
        """eos's BFS loss: reward mass on the whole frontier, not on one sampled member.

        For a row carrying a candidate set,

            loss = -log SUM_{v in candidates} softmax(logits)[v]

        so any distribution supported on the frontier is optimal. Rows without a set
        (the answering stages, where exactly one node is correct) keep cross-entropy.

        The motivation is measured, not aesthetic. A ring on a 2-arm star holds exactly
        two nodes, and drawing one of them as the label each epoch teaches the model to
        put all its mass on one -- the objective actively punishes representing both.
        That contradicts what the latents demonstrably do: the logit lens shows the
        first thought holding both frontier nodes at 0.524/0.473. It also caps measured
        accuracy at E[1/|ring|] = 0.5 however well the frontier was learned. This loss
        removes the conflict; ``set_accuracy`` in run.py measures against it honestly.

        Vectorized over the batch, where the reference implementation loops in Python
        (references/eos-upstream/coconut.py:216). Equivalent -- pinned by
        tests/test_set_loss.py -- but the ragged candidate lists are padded into a mask
        rather than indexed one row at a time.
        """
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        supervised = shift_labels != -100
        # Every row here has exactly one supervised position (the node or the answer),
        # so `argmax` finds it; rows with none are dropped below.
        has_target = supervised.any(dim=1)
        position = torch.argmax(supervised.int(), dim=1)
        rows = torch.arange(shift_logits.size(0), device=shift_logits.device)
        token_logits = shift_logits[rows, position]            # (bs, vocab)
        token_labels = shift_labels[rows, position]            # (bs,)
        log_probs = torch.log_softmax(token_logits, dim=-1)

        # Default: ordinary cross-entropy on the single gold token.
        losses = -log_probs[rows, token_labels.clamp(min=0)]

        sets = [
            (i, c) for i, c in enumerate(candidate_nodes)
            if c and len(c) > 1 and bool(has_target[i])
        ]
        if sets:
            idx = torch.tensor([i for i, _ in sets], device=log_probs.device)
            width = max(len(c) for _, c in sets)
            # Pad short rows by repeating their first member; logsumexp over duplicates
            # would double-count, so a mask zeroes the padding out to -inf instead.
            padded = torch.full((len(sets), width), 0, dtype=torch.long,
                                device=log_probs.device)
            mask = torch.zeros((len(sets), width), dtype=torch.bool,
                               device=log_probs.device)
            for r, (_, c) in enumerate(sets):
                padded[r, : len(c)] = torch.tensor(c, device=log_probs.device)
                mask[r, : len(c)] = True
            gathered = log_probs[idx].gather(1, padded)
            if self.loss_kind == "sum-of-log":
                # -(1/k) SUM log p_v: cross-entropy to the uniform distribution over the set.
                # Minimal only when every member carries mass, so early commitment is
                # penalised and the latent must hold the whole frontier.
                per_member = gathered.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
                agg = -per_member
            else:  # log-of-sum: -log SUM p_v, credits mass anywhere on the set
                agg = -torch.logsumexp(gathered.masked_fill(~mask, float("-inf")), dim=1)
            losses = losses.index_copy(0, idx, agg)

        losses = losses[has_target]
        # An all-padding batch cannot happen in training, but keep the graph intact.
        return losses.mean() if losses.numel() else logits.sum() * 0.0

    # -- forward -----------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,  # accepted and ignored, for drop-in compatibility with the original
    ) -> Outputs:
        seq_len = input_ids.shape[1]
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_(min=0)

        inputs_embeds = self.embedding(input_ids)
        first_col, counts, n_passes = self._latent_layout(input_ids)

        # Window of positions computed by the current pass. Everything before `start`
        # is already in the KV cache.
        start, end = 0, (seq_len if n_passes == 0 else first_col)
        kv_cache, logit_chunks = None, []

        # Truncated backpropagation through the thought chain. `backprop_last=k` runs every
        # pass but the last k under no_grad, so gradient reaches only the final k -- where the
        # count INCLUDES the final answer pass, so k=1 means "answer pass only" and k=n+1 is
        # full backpropagation.
        #
        # Two paths carry gradient back through the chain, and both have to be cut for this
        # to buy anything: the recurrence (thought i is the input at i+1) and the KV cache
        # (the final pass attends to keys computed in every earlier pass). Detaching the
        # hidden state alone would leave the second, so the activations would still have to be
        # retained and little would be saved. no_grad cuts both, and the memory with them --
        # unlike a `detach()`-based cut, which stops the gradient but has already recorded the
        # graph, so it keeps paying for activations it can never use.
        #
        # Note the convention when comparing k across implementations: k counts PASSES, and
        # the answer pass is one of them. An implementation that instead counts how far back
        # along the latent chain gradient travels, with the answer pass always included on
        # top, reports one less for the same window.
        #
        # What this is NOT. An earlier version of this comment claimed truncation *protects*
        # earlier stages, on the theory that stage i+1's loss can no longer reach what stage i
        # learned. The tree20 experiment refutes that: at k=5 under monotone staging the model
        # climbed to 0.98 and then collapsed to chance at the stage 7->8 transition (train loss
        # 1.09 -> 2.95) and never recovered, while the same cell under `revisit` fell back,
        # retrained the broken stage and held ~0.95. Cutting the gradient does not preserve an
        # early stage; it removes the only mechanism that was silently repairing it, which is
        # why truncation and retentive staging behave as substitutes.
        # Treat truncation as a memory optimisation that needs an explicit repair mechanism
        # alongside it, not as a retention mechanism in its own right.
        first_grad_pass = 0
        if self.backprop_last is not None:
            first_grad_pass = max(0, (n_passes + 1) - max(1, self.backprop_last))

        for pass_idx in range(n_passes + 1):
            is_final = pass_idx == n_passes
            # The final pass produces the logits the loss is taken on, so it always has grad;
            # max(1, ...) above guarantees first_grad_pass <= n_passes.
            grad_ctx = nullcontext() if pass_idx >= first_grad_pass else torch.no_grad()
            with grad_ctx:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, start:end, :],
                    # Full history: attention reads the cached keys as well as this window.
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, start:end],
                    # Passed back exactly as received -- never indexed into, so this is
                    # agnostic to the legacy tuple cache and to DynamicCache.
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=not is_final,
                )
                logit_chunks.append(outputs.logits)
                kv_cache = outputs.past_key_values
                if is_final:
                    break

                # Write thought `pass_idx` into column `end`, taken from the hidden state at
                # the preceding position. Only samples that still have a latent participate.
                rows = (counts > pass_idx).nonzero(as_tuple=True)[0]
                if rows.numel():
                    source = outputs.hidden_states[-1][rows, end - 1 - start, :]
                    inputs_embeds = inputs_embeds.index_put(
                        (rows, torch.full_like(rows, end)), source
                    )

            start, end = end, (seq_len if pass_idx + 1 == n_passes else end + 1)

        self.gen_forward_cnt += n_passes + 1

        logits = torch.cat(logit_chunks, dim=-2)
        loss = None
        if labels is not None:
            if self.loss_kind in ("log-of-sum", "sum-of-log"):
                loss = self._set_loss(logits, labels, kwargs.get("candidate_nodes") or
                                      [None] * input_ids.size(0))
            else:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                )
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits,
                       past_key_values=kv_cache)

    # -- generation ---------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 16,
        output_embedding: bool = False,
        synced_gpus: bool = False,
        **kwargs,  # accepted and ignored, for drop-in compatibility with the original
    ):
        """Greedy decoding after materializing the continuous thoughts.

        Still single-sequence (batched generation is a later step), but unlike the
        original this honours the caller's ``attention_mask`` and decodes with the KV
        cache instead of recomputing the whole prefix for every token.
        """
        if input_ids.shape[0] != 1:
            raise NotImplementedError("generate() currently supports batch_size == 1 only.")

        self.gen_forward_cnt = 0
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_(min=0)

        outputs = self.forward(input_ids, attention_mask, None, position_ids)
        kv_cache = outputs.past_key_values
        embeds = [outputs.inputs_embeds]
        tokens = input_ids[0].tolist()

        # The first token always lands, matching the original (even if it is EOS).
        next_token = int(outputs.logits[0, -1].argmax())
        tokens.append(next_token)
        n_past = input_ids.shape[1]

        for step in range(max_new_tokens - 1):
            token_embed = self.embedding(
                torch.tensor([[next_token]], device=input_ids.device)
            )
            embeds.append(token_embed)
            if step and next_token == self.eos_token_id:
                break
            outputs = self.base_causallm(
                inputs_embeds=token_embed,
                attention_mask=torch.ones(1, n_past + 1, dtype=attention_mask.dtype,
                                          device=input_ids.device),
                position_ids=torch.tensor([[n_past]], device=input_ids.device),
                past_key_values=kv_cache,
                use_cache=True,
            )
            self.gen_forward_cnt += 1
            kv_cache = outputs.past_key_values
            n_past += 1
            next_token = int(outputs.logits[0, -1].argmax())
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)

        if synced_gpus:
            # FSDP needs an identical number of forward calls on every rank. These are
            # discarded and deliberately do not touch the cache.
            dummy = embeds[-1][:, -1:, :]
            while self.gen_forward_cnt < max_new_tokens + self.max_n_latent:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=dummy)

        result = torch.tensor(tokens).view(1, -1)
        return (result, torch.cat(embeds, dim=1)) if output_embedding else result
