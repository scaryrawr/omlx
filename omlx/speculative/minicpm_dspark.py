# SPDX-License-Identifier: Apache-2.0
# Portions adapted from mlx-vlm; see ../MLX_VLM_LICENSE.txt.
"""Paired MiniCPM DSpark sampling transport over the official drafter."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel
from mlx_vlm.speculative.drafters.dspark.config import DSparkConfig

_LOADING = ContextVar("omlx_minicpm_dspark_loading", default=False)


@dataclass
class MiniCPMDSparkConfig(DSparkConfig):
    target_model_type: str = "minicpmv4_7"

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        if params.get("target_model_type", "minicpmv4_7") != "minicpmv4_7":
            raise ValueError("MiniCPM DSpark requires its explicitly paired helper")
        dflash = dict(params.get("dflash_config") or {})
        policy = dflash.get(
            "block_size_policy", params.get("block_size_policy", "fixed")
        )
        if policy != "fixed":
            raise ValueError("MiniCPM DSpark requires fixed-block verification")
        dflash["block_size_policy"] = "fixed"
        params["dflash_config"] = dflash
        return super().from_dict(params)


@dataclass(frozen=True)
class DSparkProposal:
    tokens: mx.array
    proposal_probs: mx.array | None
    confidences: mx.array | None
    block_size: int
    sample_from_anchor: bool


class MiniCPMDSparkDraftModel(DSparkDraftModel):
    def validate_target_compatibility(self, target_model):
        super().validate_target_compatibility(target_model)
        language = target_model.language_model
        if language.config.model_type != self.config.target_model_type:
            raise ValueError("MiniCPM DSpark requires its explicitly paired target")
        args = language.args
        if (args.hidden_size, args.num_hidden_layers, args.vocab_size) == (
            2048,
            40,
            248144,
        ):
            required = {
                "target_layer_ids": [3, 11, 19, 27, 35],
                "num_hidden_layers": 5,
                "head_dim": 256,
                "markov_rank": 256,
                "mask_token_id": 248076,
                "proposal_length": 15,
                "enable_confidence_head": True,
                "sample_from_anchor": True,
            }
            if any(
                getattr(self.config, name) != value for name, value in required.items()
            ):
                raise ValueError(
                    "DSpark helper does not match the published MiniCPM pairing"
                )

    def propose_block(
        self,
        last_bonus,
        hidden,
        cache,
        block_size,
        sampler,
        token_dtype=mx.int32,
        *,
        proposal_probs=None,
        sampling_key=None,
    ):
        if not 1 <= block_size <= self.config.block_size:
            raise ValueError(
                "DSpark block_size exceeds the configured verification cap"
            )
        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1).astype(token_dtype)
        )
        count = block_size - 1
        if not count:
            return DSparkProposal(
                mx.zeros((anchor.size, 0), dtype=token_dtype),
                (
                    mx.zeros((anchor.size, 0, self.config.vocab_size))
                    if proposal_probs is not None
                    else None
                ),
                (
                    mx.zeros((anchor.size, 0))
                    if self.confidence_head is not None
                    else None
                ),
                block_size,
                self.config.sample_from_anchor,
            )
        masks = mx.full(
            (anchor.size, count - int(self.config.sample_from_anchor)),
            self.config.mask_token_id,
            dtype=token_dtype,
        )
        draft_hidden = self._hidden(
            mx.concatenate([anchor[:, None], masks], axis=1), hidden, cache
        )
        if not self.config.sample_from_anchor:
            draft_hidden = draft_hidden[:, 1:]
        logits = self._logits(draft_hidden)
        tokens, probabilities = [], []
        previous = anchor
        keys = (
            mx.random.split(sampling_key, count) if sampling_key is not None else None
        )
        for index in range(count):
            step = self.markov_head.apply_step_logits(logits[:, index], previous)
            if proposal_probs is None:
                previous = sampler(step).reshape(-1)
            else:
                logprobs = step.astype(mx.float32)
                logprobs -= mx.logsumexp(logprobs, axis=-1, keepdims=True)
                probs = proposal_probs(logprobs)
                if not isinstance(probs, mx.array) or probs.shape != step.shape:
                    raise ValueError(
                        "DSpark proposal probabilities must match logits shape"
                    )
                if not bool(
                    mx.all(mx.isfinite(probs) & (probs >= 0)).item()
                ) or not bool(mx.all(mx.abs(probs.sum(axis=-1) - 1) <= 2e-5).item()):
                    raise ValueError(
                        "DSpark proposal probabilities must be finite and normalized"
                    )
                probabilities.append(probs[:, None])
                previous = mx.random.categorical(
                    mx.log(probs), key=keys[index] if keys is not None else None
                ).reshape(-1)
            tokens.append(previous[:, None])
        tokens = mx.concatenate(tokens, axis=1).astype(token_dtype)
        return DSparkProposal(
            tokens,
            mx.concatenate(probabilities, axis=1) if probabilities else None,
            self.confidence(draft_hidden, anchor, tokens),
            block_size,
            self.config.sample_from_anchor,
        )


def load_dspark_model(path: str):
    """Use upstream's strict loader with a context-local helper class override."""
    from mlx_vlm import utils

    directory = Path(path)
    MiniCPMDSparkConfig.from_dict(json.loads((directory / "config.json").read_text()))
    resolver = utils.get_model_and_args
    if not getattr(resolver, "_omlx_minicpm_dspark", False):

        def get_model_and_args(*args, **kwargs):
            if _LOADING.get():
                return (
                    SimpleNamespace(
                        Model=MiniCPMDSparkDraftModel, ModelConfig=MiniCPMDSparkConfig
                    ),
                    "dspark",
                )
            return resolver(*args, **kwargs)

        get_model_and_args._omlx_minicpm_dspark = True  # type: ignore[attr-defined]
        utils.get_model_and_args = get_model_and_args
    token = _LOADING.set(True)
    try:
        return utils.load_model(directory, strict=True)
    finally:
        _LOADING.reset(token)
