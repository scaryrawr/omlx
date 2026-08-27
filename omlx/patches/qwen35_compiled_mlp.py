# SPDX-License-Identifier: Apache-2.0
"""Compile stateless Qwen3.5-family MLP blocks for decode-shaped calls."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, ClassVar

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

logger = logging.getLogger(__name__)


class CompiledMLPBlocks:
    """Install and configure compiled stateless-MLP wrappers."""

    MAX_COMPILED_TOKENS: ClassVar[int] = 4

    @classmethod
    def enabled(cls) -> bool:
        return os.environ.get("OMLX_QWEN35_COMPILED_MLP", "1") != "0"

    @classmethod
    def _target_policies(cls) -> dict[type, type[CompiledMLPBlock]]:
        from mlx_lm.models.qwen3_next import (
            Qwen3NextMLP,
            Qwen3NextSparseMoeBlock,
        )
        from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP
        from mlx_vlm.models.qwen3_5_moe.language import (
            Qwen3_5MoeMLP,
            Qwen3_5MoeSparseMoeBlock,
        )

        for target in (Qwen3NextMLP, Qwen3NextSparseMoeBlock):
            parameters = list(inspect.signature(target.__call__).parameters)
            if parameters != ["self", "x"]:
                raise RuntimeError(
                    f"mlx-lm {target.__name__}.__call__ signature changed "
                    f"({parameters}); update the compiled MLP policy"
                )
        for target in (Qwen3_5MLP, Qwen3_5MoeMLP, Qwen3_5MoeSparseMoeBlock):
            parameters = list(inspect.signature(target.__call__).parameters)
            if parameters not in (["self", "x"], ["self", "x", "target_verify"]):
                raise RuntimeError(
                    f"mlx-vlm {target.__name__}.__call__ signature changed "
                    f"({parameters}); update the compiled MLP policy"
                )
        return {
            Qwen3NextMLP: CompiledMLPBlock,
            Qwen3NextSparseMoeBlock: CompiledMLPBlock,
            Qwen3_5MLP: CompiledTargetVerifyMLPBlock,
            Qwen3_5MoeMLP: CompiledTargetVerifyMLPBlock,
            Qwen3_5MoeSparseMoeBlock: CompiledTargetVerifyMLPBlock,
        }

    @classmethod
    def install(cls, model: Any, *, enabled: bool | None = None) -> int:
        """Wrap outermost exact-type targets after model loading is complete."""

        if enabled is None:
            enabled = cls.enabled()
        if not enabled or not mx.metal.is_available():
            return 0
        if not isinstance(model, nn.Module):
            raise TypeError(
                "CompiledMLPBlocks.install expects an mlx nn.Module, got "
                f"{type(model).__name__}"
            )

        policies = cls._target_policies()
        wrapper_prefixes = [
            name + "."
            for name, module in model.named_modules()
            if isinstance(module, CompiledMLPBlock)
        ]
        candidates = {
            name: module
            for name, module in model.named_modules()
            if type(module) in policies
            and not any(name.startswith(prefix) for prefix in wrapper_prefixes)
        }
        outermost = [
            name
            for name in candidates
            if not any(
                name.startswith(other + ".")
                for other in candidates
                if other != name
            )
        ]
        replacements = [
            (name, policies[type(candidates[name])](candidates[name]))
            for name in sorted(outermost)
        ]
        if replacements:
            if any(
                isinstance(replacement, CompiledTargetVerifyMLPBlock)
                for _, replacement in replacements
            ):
                _patch_vlm_exact_verifier()
            model.update_modules(tree_unflatten(replacements))
            logger.info(
                "Wrapped %d Qwen MLP blocks for compiled decode dispatch",
                len(replacements),
            )
        return len(replacements)


class CompiledMLPBlock(nn.Module):
    """Wrapper for stateless unary blocks called as ``block(x)``."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.train(inner.training)
        # mx.compile captures the final weight arrays. Install this wrapper
        # only after all load-time transforms; post-install weight rebinding
        # is unsupported because it would leave the trace on stale arrays.
        self._compiled = mx.compile(inner.__call__)

    def routes_compiled(self, x: mx.array) -> bool:
        return (
            not self.training
            and x.ndim == 3
            and x.shape[0] == 1
            and x.shape[1] <= CompiledMLPBlocks.MAX_COMPILED_TOKENS
            and x.dtype in (mx.float16, mx.bfloat16)
        )

    def dispatch_compiled(self, x: mx.array) -> mx.array:
        return self._compiled(x)

    def __call__(self, x: mx.array) -> mx.array:
        if self.routes_compiled(x):
            return self.dispatch_compiled(x)
        return self.inner(x)


class CompiledTargetVerifyMLPBlock(CompiledMLPBlock):
    """Wrapper that keeps mlx-vlm speculative verification eager."""

    def __init__(self, inner: nn.Module):
        super().__init__(inner)
        self._inner_accepts_target_verify = (
            "target_verify" in inspect.signature(inner.__call__).parameters
        )

    def __call__(self, x: mx.array, target_verify: bool = False) -> mx.array:
        if not target_verify and self.routes_compiled(x):
            return self.dispatch_compiled(x)
        if self._inner_accepts_target_verify:
            return self.inner(x, target_verify)
        return self.inner(x)


def _patch_vlm_exact_verifier() -> None:
    """Keep current mlx-vlm's exact verifier on the uncompiled inner block."""
    try:
        from mlx_vlm.models.qwen3_5 import language as q35
    except ImportError:
        return

    verifier = q35.LanguageModel.__call__.__globals__.get(
        "_EXACT_SPECULATIVE_VERIFIER"
    )
    if verifier is None:
        return
    verifier_cls = type(verifier)
    if getattr(verifier_cls, "_omlx_compiled_mlp_unwrap", False):
        return

    original = verifier_cls._feed_forward

    def feed_forward(self, module, x):
        if isinstance(module, CompiledTargetVerifyMLPBlock):
            module = module.inner
        return original(self, module, x)

    verifier_cls._feed_forward = feed_forward
    verifier_cls._omlx_compiled_mlp_unwrap = True


__all__ = [
    "CompiledMLPBlock",
    "CompiledMLPBlocks",
    "CompiledTargetVerifyMLPBlock",
]
