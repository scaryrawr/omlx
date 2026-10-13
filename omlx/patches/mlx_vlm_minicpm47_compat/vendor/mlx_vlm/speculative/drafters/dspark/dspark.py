from collections.abc import Callable, Mapping
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..compatibility import validate_dflash_target
from ..qwen3_dflash.dflash import DFlashDraftModel
from .config import DSparkConfig


@dataclass(frozen=True)
class DSparkProposal:
    """Tokens exclude the anchor; block_size includes it, even when sampling from it."""

    tokens: mx.array
    proposal_probs: mx.array | None
    confidences: mx.array | None
    block_size: int
    sample_from_anchor: bool


class VanillaMarkov(nn.Module):
    """Low-rank token-transition correction used by DSpark proposals."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: mx.array) -> mx.array:
        return self.markov_w1(token_ids)

    def apply_step_logits(
        self,
        logits: mx.array,
        token_ids: mx.array,
    ) -> mx.array:
        return logits + self.markov_w2(self.get_prev_embeddings(token_ids))

    def sample_block(
        self,
        base_logits: mx.array,
        *,
        first_prev_tokens: mx.array,
        sampler: Callable[[mx.array], mx.array],
        proposal_probs: Callable[[mx.array], mx.array] | None = None,
        return_probs: bool = False,
        sampling_key: mx.array | None = None,
    ) -> mx.array | tuple[mx.array, mx.array | None]:
        proposal_length = int(base_logits.shape[1])
        if proposal_length == 0:
            tokens = mx.zeros((base_logits.shape[0], 0), dtype=mx.int32)
            probs = mx.zeros_like(base_logits) if proposal_probs is not None else None
            return (tokens, probs) if return_probs else tokens

        previous_tokens = first_prev_tokens.reshape(-1)
        sampled = []
        probabilities = []
        keys = (
            mx.random.split(sampling_key, proposal_length)
            if sampling_key is not None
            else None
        )
        for step in range(proposal_length):
            step_logits = self.apply_step_logits(base_logits[:, step], previous_tokens)
            if proposal_probs is None:
                previous_tokens = sampler(step_logits).reshape(-1)
            else:
                logprobs = step_logits.astype(mx.float32)
                logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                probs = proposal_probs(logprobs)
                if not isinstance(probs, mx.array) or probs.shape != step_logits.shape:
                    raise ValueError(
                        "DSpark proposal probabilities must match logits shape."
                    )
                if not bool(
                    mx.all(mx.isfinite(probs) & (probs >= 0)).item()
                ) or not bool(
                    mx.all(mx.abs(mx.sum(probs, axis=-1) - 1) <= 2e-5).item()
                ):
                    raise ValueError(
                        "DSpark proposal probabilities must be finite and normalized."
                    )
                probabilities.append(probs[:, None])
                key = keys[step] if keys is not None else None
                previous_tokens = mx.random.categorical(mx.log(probs), key=key).reshape(
                    -1
                )
            sampled.append(previous_tokens[:, None])
        tokens = mx.concatenate(sampled, axis=1)
        probs = mx.concatenate(probabilities, axis=1) if probabilities else None
        return (tokens, probs) if return_probs else tokens


class DSparkConfidenceHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        markov_rank: int,
        with_markov: bool = True,
    ):
        super().__init__()
        self.with_markov = with_markov
        input_size = hidden_size + markov_rank if with_markov else hidden_size
        self.proj = nn.Linear(input_size, 1, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        markov_embeddings: mx.array,
    ) -> mx.array:
        features = (
            mx.concatenate([hidden_states, markov_embeddings], axis=-1)
            if self.with_markov
            else hidden_states
        )
        return mx.sigmoid(self.proj(features).squeeze(-1).astype(mx.float32))


def validate_dspark_target(config: DSparkConfig, target_model) -> None:
    validate_dflash_target(config, target_model, "DSpark")
    language_model = getattr(target_model, "language_model", target_model)
    required_type = config.target_model_type
    if required_type is not None:
        outer_config = getattr(language_model, "config", None)
        actual_type = (
            outer_config.get("model_type")
            if isinstance(outer_config, Mapping)
            else getattr(outer_config, "model_type", None)
        )
        if actual_type != required_type:
            raise ValueError(
                f"DSpark helper requires target model_type={required_type!r}."
            )
    validator = getattr(language_model, "validate_dspark_compatibility", None)
    if validator is not None:
        validator(config)


class DSparkDraftModel(DFlashDraftModel):
    """Model-agnostic DSpark proposal model over a Qwen-style draft backbone."""

    prefer_requested_block_size = False

    def __init__(self, config: DSparkConfig):
        super().__init__(config)
        self.prefer_requested_block_size = config.block_size_policy == "fixed"
        self.dflash_initial_block_size = config.dflash_initial_block_size
        self.markov_head = VanillaMarkov(config.vocab_size, config.markov_rank)
        self.confidence_head: DSparkConfidenceHead | None = (
            DSparkConfidenceHead(
                config.hidden_size,
                config.markov_rank,
                with_markov=config.confidence_head_with_markov,
            )
            if config.enable_confidence_head
            else None
        )

    def validate_target_compatibility(self, target_model) -> None:
        validate_dspark_target(self.config, target_model)

    def choose_initial_block_size(
        self, context_length: int, requested_block_size: int
    ) -> int | None:
        window = self.config.sliding_window
        if window is not None and context_length <= window:
            return min(6, requested_block_size)
        return self.dflash_initial_block_size

    def choose_block_ceiling(
        self, context_length: int, requested_block_size: int
    ) -> int:
        window = self.config.sliding_window
        if window is not None and context_length <= window:
            return min(6, requested_block_size)
        if window is not None:
            return min(4, requested_block_size)
        return requested_block_size

    def bind(self, target_model) -> "DSparkDraftModel":
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        return self

    def confidence(
        self,
        draft_hidden: mx.array,
        anchor_tokens: mx.array,
        draft_tokens: mx.array,
    ) -> mx.array | None:
        if self.confidence_head is None:
            return None
        previous = mx.concatenate(
            [anchor_tokens.reshape(-1, 1), draft_tokens[:, :-1]], axis=1
        )
        markov_embeddings = self.markov_head.get_prev_embeddings(previous)
        return self.confidence_head(draft_hidden, markov_embeddings)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        return self.propose_block(
            last_bonus, hidden, cache, block_size, sampler, token_dtype
        ).tokens

    def propose_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
        *,
        proposal_probs: Callable[[mx.array], mx.array] | None = None,
        sampling_key: mx.array | None = None,
    ) -> DSparkProposal:
        """Propose sequentially; a probability callback owns the sampling law.

        The callback maps normalized Markov-conditioned log probabilities to
        normalized sampling probabilities (including temperature/filters).
        Without it the legacy sampler owns sampling, so its law is unknown and
        ``proposal_probs`` in the result is None, never a guessed softmax.
        Confidence is observational: it does not truncate the fixed block.
        """
        if not 1 <= block_size <= self.config.block_size:
            raise ValueError(
                "DSpark block_size exceeds the configured verification cap."
            )
        proposal_length = int(block_size) - 1
        if proposal_length <= 0:
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return DSparkProposal(
                tokens=mx.zeros((batch, 0), dtype=token_dtype),
                proposal_probs=(
                    mx.zeros((batch, 0, self.config.vocab_size))
                    if proposal_probs is not None
                    else None
                ),
                confidences=(
                    mx.zeros((batch, 0)) if self.confidence_head is not None else None
                ),
                block_size=block_size,
                sample_from_anchor=self.config.sample_from_anchor,
            )

        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1).astype(token_dtype)
        )
        mask_count = proposal_length - int(self.config.sample_from_anchor)
        masks = mx.full(
            (anchor.shape[0], mask_count),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        draft_hidden = self._hidden(draft_inputs, hidden, cache)
        if not self.config.sample_from_anchor:
            draft_hidden = draft_hidden[:, 1:]
        base_logits = self._logits(draft_hidden)
        tokens, probs = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            sampler=sampler,
            proposal_probs=proposal_probs,
            return_probs=True,
            sampling_key=sampling_key,
        )
        return DSparkProposal(
            tokens=tokens.astype(token_dtype),
            proposal_probs=probs,
            confidences=self.confidence(draft_hidden, anchor, tokens),
            block_size=block_size,
            sample_from_anchor=self.config.sample_from_anchor,
        )

    def draft_block_greedy(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        return self.draft_block(
            last_bonus,
            hidden,
            cache,
            block_size,
            sampler,
            token_dtype,
        )

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        normalized = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in normalized:
                raise ValueError(
                    f"Duplicate DSpark weight key after sanitization: {key}"
                )
            normalized[key] = value
        if "embed_tokens.weight" in normalized and self.embed_tokens is None:
            self.embed_tokens = nn.Embedding(
                self.config.vocab_size, self.config.hidden_size
            )
        return normalized


Model = DSparkDraftModel


__all__ = [
    "DSparkConfidenceHead",
    "DSparkDraftModel",
    "DSparkProposal",
    "Model",
    "VanillaMarkov",
    "validate_dspark_target",
]
