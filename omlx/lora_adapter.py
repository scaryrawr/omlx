# SPDX-License-Identifier: Apache-2.0
"""PEFT LoRA adapters served as extra model entries.

An adapter directory is read from its config and its safetensors header only, bound to
a local base model, and applied when an engine loads by wrapping the base's linear
layers in LoraLinear. Parsing and resolution are pure and never touch MLX arrays.

mlx_vlm.trainer and mlx_lm.tuner are not imported: each costs 1 to 2 s.
omlx.model_discovery is imported for typing only and never imports this module at
runtime, so the two cannot form an import cycle.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Protocol, TypeGuard

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import switch_layers as lm_switch_layers

from .exceptions import AdapterError, AdapterShapeError, UnsupportedAdapterError
from .patches.inkling2 import INKLING2_MODEL_TYPES

if TYPE_CHECKING:
    from .model_discovery import DiscoveredModel

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
_PICKLED_WEIGHTS_NAME = "adapter_model.bin"

_MAX_CONFIG_BYTES = 1 << 20
_MAX_HEADER_BYTES = 100 << 20
_DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2}
_TENSOR_KEY = re.compile(
    r"base_model\.model\.(?P<path>.+)\.lora_(?P<half>[AB])\.weight"
)
_LANGUAGE_PREFIX = r"(?:model\.)?(?:language_model\.(?:model\.)?)?"
_LAYER_TAIL = re.compile(rf"{_LANGUAGE_PREFIX}(layers\.\d+\..+)")
_INKLING2_PROJECTIONS = {"self_attn.o_proj": "attn.o_proj"}
_ROUTED_DOWN = re.compile(
    rf"{_LANGUAGE_PREFIX}layers\.\d+\.(?:mlp\.)?experts\.down_proj"
)
_UNSUPPORTED_WHEN_TRUTHY = (
    "use_dora",
    "use_qalora",
    "modules_to_save",
    "rank_pattern",
    "alpha_pattern",
    "layer_replication",
    "fan_in_fan_out",
)
_BASE_MODEL_TYPES = frozenset({"llm", "vlm"})
_BASE_ENGINE_TYPES = frozenset({"batched", "vlm"})
_MERGE_FIX = "Merge it with peft merge_and_unload and serve the merged model."
_CHANGED_AFTER_DISCOVERY = (
    "The adapter files changed after discovery. Reload the model list."
)


@dataclass(frozen=True)
class LoraConfig:
    base_ref: str  # adapter_config.base_model_name_or_path. A match key only.
    rank: int
    alpha: float
    rslora: bool
    target_parameters: tuple[str, ...] = ()

    @property
    def scale(self) -> float:
        """alpha / sqrt(rank) when rslora else alpha / rank (PEFT)."""
        return self.alpha / (math.sqrt(self.rank) if self.rslora else self.rank)


@dataclass(frozen=True)
class LoraModule:
    """One targeted module, read from the safetensors header only."""

    hf_path: str  # "model.language_model.layers.3.mlp.gate_proj"
    in_features: int  # lora_A.weight shape [r, in]
    out_features: int  # lora_B.weight shape [out, r]


@dataclass(frozen=True)
class _RoutedModule(LoraModule):
    experts: int

    @property
    def tensor_stem(self) -> str:
        return self.hf_path.removesuffix(".down_proj")

    @property
    def runtime_path(self) -> str:
        if self.tensor_stem.endswith(".mlp.experts"):
            return self.tensor_stem.removesuffix(".experts") + ".switch_mlp.down_proj"
        return self.tensor_stem + ".switch_glu.down_proj"


@dataclass(frozen=True)
class AdapterSource:
    """A structurally valid adapter directory. Not yet bound to a base."""

    public_id: str  # directory basename; never contains ":" (profiles own ":")
    path: Path
    config: LoraConfig
    modules: tuple[LoraModule, ...]
    weights_path: Path
    weight_bytes: int  # file size == resident bytes (tensors stay as shipped)
    fingerprint: str  # sha256(config bytes + weights size + weights mtime_ns)

    @property
    def has_routed_targets(self) -> bool:
        return bool(self.config.target_parameters)


@dataclass(frozen=True)
class BaseRef:
    model_id: str
    path: Path  # physical weights directory of the base


@dataclass(frozen=True)
class AdapterBinding:
    """Adapter plus resolved base. The only adapter type an engine ever receives.

    There is no Optional base. An unresolved adapter is an AdapterIssue and never
    becomes an entry, so no engine can see a half-bound adapter.
    """

    source: AdapterSource
    base: BaseRef
    cache_revision: str

    def as_status(self) -> dict[str, Any]:
        """{"base_model_id", "rank", "alpha", "weight_bytes"} for get_status() rows."""
        return {
            "base_model_id": self.base.model_id,
            "rank": self.source.config.rank,
            "alpha": self.source.config.alpha,
            "weight_bytes": self.source.weight_bytes,
        }


def bind_adapter(source: AdapterSource, base: BaseRef) -> AdapterBinding:
    """The one constructor of AdapterBinding. A pure function with no file reads.

    cache_revision changes with a retrained adapter or a re-bound base path, so stale
    SSD KV and vision-feature blocks stop matching. An in-place change of the base
    weights is the exposure plain models already have.
    """
    digest = hashlib.sha256(f"{source.fingerprint}|{base.path}".encode()).hexdigest()
    return AdapterBinding(source, base, f"lora-{digest[:12]}")


class IssueKind(enum.Enum):
    UNSUPPORTED = "unsupported"  # valid PEFT, feature we do not apply
    MALFORMED = "malformed"  # unreadable, missing weights, bad tensors or shapes
    BASE_MISSING = "base_missing"
    BASE_AMBIGUOUS = "base_ambiguous"
    OVERRIDE_INVALID = "override_invalid"  # persisted override names no usable base
    ID_COLLISION = "id_collision"  # id taken by a real model or an earlier adapter


@dataclass(frozen=True)
class AdapterIssue:
    """Why an adapter directory is not served. It never becomes an entry."""

    public_id: str
    path: Path
    kind: IssueKind
    detail: str  # operator-facing: says what to download, pin or merge
    candidates: tuple[str, ...] = ()  # BASE_AMBIGUOUS: the model ids to choose from


class ModelLike(Protocol):
    """The slice of DiscoveredModel and EngineEntry that base matching reads."""

    @property
    def model_id(self) -> str: ...
    @property
    def model_path(self) -> str: ...
    @property
    def source_repo_id(self) -> str | None: ...
    @property
    def is_helper(self) -> bool: ...
    @property
    def model_type(self) -> str: ...
    @property
    def engine_type(self) -> str: ...
    @property
    def model_format(self) -> str | None: ...
    @property
    def adapter(self) -> object | None: ...


@dataclass(frozen=True)
class BaseCandidate:
    """What resolution needs to know about a local base, nothing more."""

    model_id: str
    path: Path
    repo_ids: frozenset[str]  # lowercased: source_repo_id and "<parent>/<name>" of path


@dataclass(frozen=True)
class AdapterDiscovery:
    """Derived entries by public id, plus the adapters that could not be served."""

    entries: dict[str, DiscoveredModel]
    issues: tuple[AdapterIssue, ...]


def _clip(value: object, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _is_int(value: object) -> TypeGuard[int]:
    """True for an int that is not a bool; JSON true must not pass as a count."""
    return isinstance(value, int) and not isinstance(value, bool)


def _unsupported(feature: str, problem: str | None = None) -> UnsupportedAdapterError:
    problem = problem or f"The adapter uses {feature}, which oMLX does not apply."
    return UnsupportedAdapterError(f"{problem} {_MERGE_FIX}", feature=feature)


def _alpha_from(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        alpha = math.nan
    else:
        try:
            alpha = float(value)
        except OverflowError:
            alpha = math.inf
    if not math.isfinite(alpha) or alpha <= 0:
        raise AdapterError(
            f"{ADAPTER_CONFIG_NAME} needs lora_alpha to be a positive finite number."
        )
    return alpha


def parse_adapter_config(raw: bytes) -> LoraConfig:
    """adapter_config.json bytes -> LoraConfig. Shared with the downloader card.

    Raises UnsupportedAdapterError for a valid config oMLX does not apply, and
    AdapterError for one that is malformed. base_model_name_or_path is a match key
    only. It is never opened or fetched.
    """
    if len(raw) > _MAX_CONFIG_BYTES:
        raise AdapterError(f"{ADAPTER_CONFIG_NAME} is larger than 1 MiB.")
    try:
        document = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise AdapterError(f"{ADAPTER_CONFIG_NAME} is not valid JSON.") from error
    if not isinstance(document, dict):
        raise AdapterError(f"{ADAPTER_CONFIG_NAME} must hold a JSON object.")

    peft_type = document.get("peft_type")
    if not isinstance(peft_type, str):
        raise AdapterError(f"{ADAPTER_CONFIG_NAME} has no peft_type.")
    if peft_type != "LORA":
        raise _unsupported(
            "peft_type",
            f"The adapter has peft_type {_clip(peft_type)}, "
            "and oMLX applies only LORA.",
        )
    for key in _UNSUPPORTED_WHEN_TRUTHY:
        if document.get(key):
            raise _unsupported(key)
    if document.get("alora_invocation_tokens") is not None:
        raise UnsupportedAdapterError(
            "oMLX does not support aLoRA's token-triggered activation.",
            feature="alora_invocation_tokens",
        )
    if document.get("bias") not in (None, "none"):
        raise _unsupported("bias")

    initializer = document.get("init_lora_weights", True)
    if not isinstance(initializer, bool) and initializer not in (
        "gaussian",
        "eva",
        "orthogonal",
        "mica",
        "nora",
    ):
        raise UnsupportedAdapterError(
            f"oMLX cannot apply init_lora_weights={_clip(initializer)} "
            "as a standalone delta on the original base. Use a standard "
            "LoRA export that does not require changing the base weights.",
            feature="init_lora_weights",
        )

    targets = document.get("target_parameters")
    if targets is None:
        targets = []
    if not isinstance(targets, list) or not all(
        isinstance(path, str) and _ROUTED_DOWN.fullmatch(path) for path in targets
    ):
        raise _unsupported("target_parameters")
    if len(set(targets)) != len(targets):
        raise AdapterError("target_parameters contains duplicate expert projections.")

    base_ref = document.get("base_model_name_or_path")
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise AdapterError(
            f"{ADAPTER_CONFIG_NAME} needs base_model_name_or_path "
            "to be a non-empty string."
        )
    rank = document.get("r")
    if not _is_int(rank) or rank <= 0:
        raise AdapterError(f"{ADAPTER_CONFIG_NAME} needs r to be a positive integer.")
    return LoraConfig(
        base_ref=base_ref,
        rank=rank,
        alpha=_alpha_from(document.get("lora_alpha")),
        rslora=bool(document.get("use_rslora")),
        target_parameters=tuple(targets),
    )


def _read_config(path: Path) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read(_MAX_CONFIG_BYTES + 1)
    except FileNotFoundError as error:
        raise AdapterError(f"{path.name} is missing.") from error
    except OSError as error:
        raise AdapterError(
            f"Cannot read {path.name}: {error.strerror or error}."
        ) from error


def _read_header(handle: BinaryIO) -> tuple[dict[str, Any], int]:
    """The safetensors JSON header and the offset where tensor data starts."""
    prefix = handle.read(8)
    header_len = int.from_bytes(prefix, "little")
    if len(prefix) < 8 or header_len > _MAX_HEADER_BYTES:
        raise AdapterError(
            f"{ADAPTER_WEIGHTS_NAME} is truncated or has an invalid header. "
            "Download it again."
        )
    raw = handle.read(header_len)
    if len(raw) < header_len:
        raise AdapterError(f"{ADAPTER_WEIGHTS_NAME} is truncated. Download it again.")
    try:
        header = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise AdapterError(
            f"{ADAPTER_WEIGHTS_NAME} has an unreadable header."
        ) from error
    if not isinstance(header, dict):
        raise AdapterError(f"{ADAPTER_WEIGHTS_NAME} has an unreadable header.")
    return header, 8 + header_len


def _read_weights_header(path: Path) -> tuple[dict[str, Any], int, int, int]:
    """(header, data offset, file size, mtime_ns), all from one open handle."""
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            header, data_start = _read_header(handle)
    except FileNotFoundError as error:
        if os.path.isfile(path.with_name(_PICKLED_WEIGHTS_NAME)):
            raise UnsupportedAdapterError(
                f"The adapter ships {_PICKLED_WEIGHTS_NAME}, and oMLX never loads "
                f"pickled weights. Re-save it as {ADAPTER_WEIGHTS_NAME}.",
                feature="weights_format",
            ) from error
        raise AdapterError(f"{path.name} is missing.") from error
    except OSError as error:
        raise AdapterError(
            f"Cannot read {path.name}: {error.strerror or error}."
        ) from error
    return header, data_start, stat.st_size, stat.st_mtime_ns


def _tensor_shape(
    name: str, entry: object, data_start: int, file_size: int
) -> tuple[int, ...]:
    if not isinstance(entry, dict):
        raise AdapterError(f"Tensor {_clip(name)} has no description.")
    dtype = entry.get("dtype")
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise AdapterError(
            f"Tensor {_clip(name)} has dtype {_clip(dtype)}. Use F32, BF16 or F16."
        )
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(shape, list)
        or not all(_is_int(dim) and dim >= 0 for dim in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(_is_int(offset) for offset in offsets)
    ):
        raise AdapterError(f"Tensor {_clip(name)} has a malformed shape or offsets.")
    begin, end = offsets
    # A truncated download must fail here, not when the engine loads the weights.
    if not 0 <= begin <= end or data_start + end > file_size:
        raise AdapterError(
            f"Tensor {_clip(name)} reaches past the end of {ADAPTER_WEIGHTS_NAME}. "
            "Download it again."
        )
    if end - begin != _DTYPE_BYTES[dtype] * math.prod(shape):
        raise AdapterError(
            f"Tensor {_clip(name)} holds data that does not fit its shape."
        )
    return tuple(shape)


def _modules_from_header(
    header: dict[str, Any], data_start: int, file_size: int, config: LoraConfig
) -> tuple[LoraModule, ...]:
    halves: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        match = _TENSOR_KEY.fullmatch(name)
        if match is None:
            raise AdapterError(
                f"Unexpected tensor {_clip(name)}. "
                "Only LoRA A and B weights are supported."
            )
        shape = _tensor_shape(name, entry, data_start, file_size)
        halves.setdefault(match["path"], {})[match["half"]] = shape

    routed = {
        path.removesuffix(".down_proj"): path for path in config.target_parameters
    }
    for stem, path in routed.items():
        if stem not in halves:
            raise AdapterError(
                f"Expert projection {_clip(path)} needs lora_A and lora_B."
            )

    rank = config.rank
    modules: list[LoraModule] = []
    for hf_path in sorted(halves):
        pair = halves[hf_path]
        if pair.keys() != {"A", "B"}:
            raise AdapterError(f"Module {_clip(hf_path)} needs both lora_A and lora_B.")
        shape_a, shape_b = pair["A"], pair["B"]
        if len(shape_a) != 2 or len(shape_b) != 2:
            raise AdapterError(f"Module {_clip(hf_path)} needs 2-D LoRA weights.")
        if hf_path in routed:
            experts, remainder = divmod(shape_a[0], rank)
            if not experts or remainder or shape_b[1] != experts * rank:
                raise AdapterError(
                    f"Expert projection {_clip(routed[hf_path])} needs matching "
                    f"expert counts in lora_A and lora_B at rank {rank}."
                )
            modules.append(
                _RoutedModule(routed[hf_path], shape_a[1], shape_b[0], experts)
            )
            continue
        if _ROUTED_DOWN.fullmatch(hf_path + ".down_proj"):
            raise AdapterError(
                f"Expert container {_clip(hf_path)} has no declared target."
            )
        if shape_a[0] != rank or shape_b[1] != rank:
            raise AdapterError(
                f"Module {_clip(hf_path)} has rank {shape_a[0]} and {shape_b[1]}, "
                f"but {ADAPTER_CONFIG_NAME} says r={rank}."
            )
        modules.append(
            LoraModule(hf_path, in_features=shape_a[1], out_features=shape_b[0])
        )
    if not modules:
        raise AdapterError(f"{ADAPTER_WEIGHTS_NAME} holds no LoRA weights.")
    return tuple(modules)


def _fingerprint(config_bytes: bytes, weight_bytes: int, mtime_ns: int) -> str:
    """The identity of an adapter's files. Tensor data is not hashed."""
    stamp = f"{len(config_bytes)}|{weight_bytes}|{mtime_ns}|".encode()
    return hashlib.sha256(stamp + config_bytes).hexdigest()


def _parse_adapter(adapter_dir: Path) -> AdapterSource:
    config_bytes = _read_config(adapter_dir / ADAPTER_CONFIG_NAME)
    config = parse_adapter_config(config_bytes)
    weights_path = adapter_dir / ADAPTER_WEIGHTS_NAME
    header, data_start, weight_bytes, mtime_ns = _read_weights_header(weights_path)
    modules = _modules_from_header(header, data_start, weight_bytes, config)
    return AdapterSource(
        public_id=adapter_dir.name,
        path=adapter_dir,
        config=config,
        modules=modules,
        weights_path=weights_path,
        weight_bytes=weight_bytes,
        fingerprint=_fingerprint(config_bytes, weight_bytes, mtime_ns),
    )


def parse_adapter(adapter_dir: Path) -> AdapterSource:
    """Read the config and the safetensors header. No tensor data, no pickle, no code.

    Raises UnsupportedAdapterError for a PEFT feature or weights format oMLX does not
    apply, and AdapterError for anything unreadable, missing, truncated or misshapen.
    """
    try:
        return _parse_adapter(adapter_dir)
    except AdapterError as error:
        error.details.setdefault("adapter", str(adapter_dir))
        raise


def base_candidates(models: Iterable[ModelLike]) -> tuple[BaseCandidate, ...]:
    """Chat-capable local bases, in input order.

    A base is model_type llm or vlm on the batched or vlm engine, not a helper, not a
    GGUF folder (an external engine serves those) and not itself an adapter entry.
    repo_ids holds the lowercased source_repo_id (when set) and "<parent>/<name>" of
    the model directory, so a base downloaded as owner/name matches an adapter's
    base_model_name_or_path exactly.
    """
    candidates = []
    for model in models:
        if (
            model.model_type not in _BASE_MODEL_TYPES
            or model.engine_type not in _BASE_ENGINE_TYPES
            or model.is_helper
            or model.model_format is not None
            or model.adapter is not None
        ):
            continue
        path = Path(model.model_path)
        repo_ids = {f"{path.parent.name}/{path.name}".lower()}
        if model.source_repo_id:
            repo_ids.add(model.source_repo_id.lower())
        candidates.append(BaseCandidate(model.model_id, path, frozenset(repo_ids)))
    return tuple(candidates)


def match_bases(
    base_ref: str, candidates: Iterable[BaseCandidate]
) -> tuple[BaseCandidate, ...]:
    """Exact tier first, family tier second. Shared by resolution and the downloader.

    Exact: candidates whose repo_ids contain base_ref, ignoring case. Any exact hit
    wins outright.
    Family: the last path segment of base_ref is the stem, and a candidate's model id
    must equal it or continue it after a "-" or "_".
    """
    candidates = tuple(candidates)
    wanted = base_ref.lower()
    exact = tuple(c for c in candidates if wanted in c.repo_ids)
    if exact:
        return exact
    stem = wanted.rsplit("/", 1)[-1]
    if not stem:
        return ()
    # The separator keeps "qwen3.8-27b" from matching "qwen3.8-270b".
    continuations = (f"{stem}-", f"{stem}_")
    return tuple(
        c
        for c in candidates
        if c.model_id.lower() == stem or c.model_id.lower().startswith(continuations)
    )


def resolve_adapters(
    sources: Iterable[AdapterSource | AdapterIssue],
    candidates: Iterable[BaseCandidate],
    overrides: Mapping[str, str],
    taken_ids: Collection[str],
) -> tuple[tuple[AdapterBinding, ...], tuple[AdapterIssue, ...]]:
    candidates = tuple(candidates)
    by_id = {candidate.model_id: candidate for candidate in candidates}
    reserved: set[str] = set()
    bindings: list[AdapterBinding] = []
    issues: list[AdapterIssue] = []

    def reject(
        source: AdapterSource,
        kind: IssueKind,
        detail: str,
        choices: tuple[str, ...] = (),
    ) -> None:
        issues.append(
            AdapterIssue(source.public_id, source.path, kind, detail, choices)
        )

    for source in sources:
        public_id = source.public_id
        if isinstance(source, AdapterIssue):
            reserved.add(public_id)
            continue
        pin = f"PUT /admin/api/adapters/{public_id}/base"
        if ":" in public_id:
            reject(
                source,
                IssueKind.ID_COLLISION,
                f"Adapter id {public_id!r} contains ':', which model profiles use. "
                "Rename the adapter directory.",
            )
            continue
        if public_id in taken_ids or public_id in reserved:
            reject(
                source,
                IssueKind.ID_COLLISION,
                f"Adapter id {public_id!r} is already used by a model or an adapter. "
                "Rename the adapter directory.",
            )
            continue
        reserved.add(public_id)

        pinned = overrides.get(public_id)
        if pinned is not None:
            base = by_id.get(pinned)
            if base is None:
                reject(
                    source,
                    IssueKind.OVERRIDE_INVALID,
                    f"The pinned base {pinned!r} is not a local chat model. "
                    f"Pin another base or clear the pin with {pin}.",
                )
                continue
            matches: tuple[BaseCandidate, ...] = (base,)
        else:
            matches = match_bases(source.config.base_ref, candidates)

        if len(matches) == 1:
            base = matches[0]
            bindings.append(bind_adapter(source, BaseRef(base.model_id, base.path)))
        elif not matches:
            reject(
                source,
                IssueKind.BASE_MISSING,
                f"No local model matches the base {source.config.base_ref!r}. "
                f"Download it, or pin a local base with {pin}.",
            )
        else:
            names = tuple(match.model_id for match in matches)
            reject(
                source,
                IssueKind.BASE_AMBIGUOUS,
                f"Several local models match the base {source.config.base_ref!r}: "
                f"{', '.join(names)}. Pin one with {pin}.",
                names,
            )
    return tuple(bindings), tuple(issues)


def derive_entry(base: DiscoveredModel, binding: AdapterBinding) -> DiscoveredModel:
    """The model entry an adapter adds, copied from its base.

    model_path stays the base directory, so the physical read sites need no edit. The
    adapter's weights add to the size estimates. thinking_default becomes False when the
    base knows it, because an adapter tuned for a harness does not think by default.
    """
    weight_bytes = binding.source.weight_bytes
    return dataclasses.replace(
        base,
        model_id=binding.source.public_id,
        adapter=binding,
        source_type="local",
        source_repo_id=None,
        is_helper=False,
        estimated_size=base.estimated_size + weight_bytes,
        text_only_size=base.text_only_size + weight_bytes if base.text_only_size else 0,
        thinking_default=False if base.thinking_default is not None else None,
    )


def discover_adapters(
    adapter_dirs: Iterable[Path],
    discovered: Mapping[str, DiscoveredModel],
    base_overrides: Mapping[str, str],
) -> AdapterDiscovery:
    """Parse each directory, bind it to a discovered base, and derive its entry.

    A directory that cannot be served becomes an issue, never an exception. Parse
    issues come first, then resolution issues. Entries never override a real model id.
    """
    sources: list[AdapterSource | AdapterIssue] = []
    parse_issues: list[AdapterIssue] = []
    for adapter_dir in adapter_dirs:
        try:
            sources.append(parse_adapter(adapter_dir))
        except AdapterError as error:
            kind = (
                IssueKind.UNSUPPORTED
                if isinstance(error, UnsupportedAdapterError)
                else IssueKind.MALFORMED
            )
            issue = AdapterIssue(adapter_dir.name, adapter_dir, kind, error.message)
            parse_issues.append(issue)
            sources.append(issue)

    bindings, resolve_issues = resolve_adapters(
        sources, base_candidates(discovered.values()), base_overrides, set(discovered)
    )
    bases = {model.model_id: model for model in discovered.values()}
    entries = {
        binding.source.public_id: derive_entry(bases[binding.base.model_id], binding)
        for binding in bindings
    }
    return AdapterDiscovery(entries, (*parse_issues, *resolve_issues))


class _LoraProjection(nn.Module):
    def __init__(
        self,
        linear: nn.Module,
        lora_a: mx.array,
        lora_b: mx.array,
        scale: float,
        revision: str,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scale = float(scale)
        self.revision = revision


class LoraLinear(_LoraProjection):
    """y = linear(x) + (scale * ((x @ lora_a) @ lora_b)).astype(x.dtype)

    The math of mlx_vlm.trainer.peft.lora_layers.LoRALinear.__call__ (mlx_lm.tuner.lora
    has the same class) without dropout, and without importing either trainer package.
    lora_a is [in, r] and lora_b is [r, out], kept in the dtype the adapter shipped in.
    It is not a subclass of nn.QuantizedLinear and exposes no .weight, so code that
    reads linear attributes directly fails loudly and exact-type-gated patches skip the
    wrapper. revision is a plain str attribute, not a parameter.
    """

    def __call__(self, x: mx.array) -> mx.array:
        y: mx.array = self.linear(x)
        delta = (x @ self.lora_a) @ self.lora_b
        return y + (self.scale * delta).astype(x.dtype)


class _LoraSwitchLinear(_LoraProjection):
    def __call__(
        self, x: mx.array, indices: mx.array, sorted_indices: bool = False
    ) -> mx.array:
        y: mx.array = self.linear(x, indices, sorted_indices=sorted_indices)
        h = mx.gather_mm(
            x, self.lora_a, rhs_indices=indices, sorted_indices=sorted_indices
        )
        delta = mx.gather_mm(
            h, self.lora_b, rhs_indices=indices, sorted_indices=sorted_indices
        )
        return y + (self.scale * delta).astype(y.dtype)


@dataclass(frozen=True)
class _Replacement:
    hf_path: str
    parent: Any
    key: str | int
    target: nn.Module
    lora_a: mx.array
    lora_b: mx.array


def _tensor_name(hf_path: str, half: str) -> str:
    return f"base_model.model.{hf_path}.lora_{half}.weight"


def _target_name(
    hf_path: str,
    named: Mapping[str, nn.Module],
    adapter: Path,
    model_type: str | None,
) -> str:
    """The name under the language root that an adapter's HF module path points at."""
    match = _LAYER_TAIL.fullmatch(hf_path)
    if match is None:
        raise AdapterShapeError(
            f"{hf_path} is not inside a supported numbered language layer.",
            adapter,
        )
    tail = match.group(1)
    found = [name for name in (f"model.{tail}", tail) if name in named]
    if not found and model_type in INKLING2_MODEL_TYPES:
        _, index, projection = tail.split(".", 2)
        runtime_projection = _INKLING2_PROJECTIONS.get(projection)
        if runtime_projection is not None:
            tail = f"layers.{index}.{runtime_projection}"
            found = [name for name in (f"model.{tail}", tail) if name in named]
    if len(found) != 1:
        problem = "is not in" if not found else "matches two modules of"
        raise AdapterShapeError(
            f"{hf_path} {problem} the base model. Check that the adapter was trained "
            "for this base.",
            adapter,
        )
    return found[0]


def _linear_features(name: str, target: nn.Module, adapter: Path) -> tuple[int, int]:
    """(in_features, out_features) of a dense or quantized linear.

    Any other module type is an AdapterShapeError. oMLX cannot know how a custom
    wrapper applies its weights, so it fails closed.
    """
    from .oqh.modules import OQH_DENSE

    if isinstance(target, nn.QuantizedLinear):
        # scales.shape[-1] * group_size is exact for every bit width and mode.
        return target.scales.shape[-1] * target.group_size, target.weight.shape[0]
    if isinstance(target, nn.Linear):
        return target.weight.shape[-1], target.weight.shape[0]
    if type(target) in OQH_DENSE:
        return target.input_dims, target.output_dims
    raise AdapterShapeError(
        f"Base module {name} is a {type(target).__name__}, not a Linear or "
        "QuantizedLinear or oQh linear, so oMLX will not wrap it.",
        adapter,
    )


def _locate(root: Any, name: str) -> tuple[Any, str | int]:
    """The container holding the module at a dotted name, and its key in it."""
    *parents, last = name.split(".")
    node = root
    for part in parents:
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, Mapping):
            node = node[part]
        else:
            node = getattr(node, part)
    return node, int(last) if isinstance(node, list) else last


def _plan_replacement(
    parent: Any,
    key: str | int,
    name: str,
    target: nn.Module,
    module: LoraModule,
    weights: Mapping[str, mx.array],
    source: AdapterSource,
) -> _Replacement:
    stem = module.hf_path
    rank = source.config.rank
    if isinstance(module, _RoutedModule):
        from mlx_vlm.models import switch_layers as vlm_switch_layers

        from .oqh.modules import OQH_SWITCH
        from .oqh.runtime import _FusedSwitchGLU

        parent_type = type(parent)
        if issubclass(parent_type, _FusedSwitchGLU):
            parent_type = parent_type.__bases__[1]
        supported = any(
            parent_type is layers.SwitchGLU
            and type(target)
            in (layers.SwitchLinear, layers.QuantizedSwitchLinear, *OQH_SWITCH)
            for layers in (lm_switch_layers, vlm_switch_layers)
        )
        if not supported:
            raise AdapterShapeError(
                f"Base module {name} is not a supported routed expert projection.",
                source.path,
            )
        if target.num_experts != module.experts:
            raise AdapterShapeError(
                f"{module.hf_path} has {module.experts} experts, but base module "
                f"{name} has {target.num_experts}.",
                source.path,
            )
        in_features, out_features = target.input_dims, target.output_dims
        stem = module.tensor_stem
        packed_rank = rank * module.experts
    else:
        in_features, out_features = _linear_features(name, target, source.path)
        packed_rank = rank
    if (in_features, out_features) != (module.in_features, module.out_features):
        raise AdapterShapeError(
            f"{module.hf_path} was trained for {module.in_features} inputs and "
            f"{module.out_features} outputs, but base module {name} has "
            f"{in_features} and {out_features}. Check that the adapter was trained "
            "for this base.",
            source.path,
        )
    lora_a = weights.get(_tensor_name(stem, "A"))
    lora_b = weights.get(_tensor_name(stem, "B"))
    if (
        lora_a is None
        or lora_b is None
        or tuple(lora_a.shape) != (packed_rank, in_features)
        or tuple(lora_b.shape) != (out_features, packed_rank)
    ):
        raise AdapterShapeError(
            f"The weights of {module.hf_path} in {ADAPTER_WEIGHTS_NAME} differ from "
            "the adapter that was discovered. Reload the model list.",
            source.path,
        )
    if isinstance(module, _RoutedModule):
        a = lora_a.reshape(module.experts, rank, in_features).transpose(0, 2, 1)
        # PEFT packs B in [out, rank, expert] order, not [out, expert, rank].
        b = lora_b.reshape(out_features, rank, module.experts).transpose(2, 1, 0)
        return _Replacement(module.hf_path, parent, key, target, a, b)
    return _Replacement(module.hf_path, parent, key, target, lora_a.T, lora_b.T)


def _install(parent: Any, key: str | int, wrapper: nn.Module) -> None:
    if isinstance(parent, nn.Module):
        setattr(parent, str(key), wrapper)
    else:
        parent[key] = wrapper


def _require_unchanged(source: AdapterSource) -> None:
    """Refuse an adapter whose files no longer match what discovery fingerprinted.

    cache_revision derives from that fingerprint. Tensors read from replaced files
    would otherwise be cached under the identity of the files they replaced.
    """
    try:
        config_bytes = _read_config(source.path / ADAPTER_CONFIG_NAME)
        stat = source.weights_path.stat()
    except (AdapterError, OSError) as error:
        raise AdapterShapeError(_CHANGED_AFTER_DISCOVERY, source.path) from error
    now = _fingerprint(config_bytes, stat.st_size, stat.st_mtime_ns)
    if now != source.fingerprint:
        raise AdapterShapeError(_CHANGED_AFTER_DISCOVERY, source.path)


def attach_lora(model: Any, binding: AdapterBinding) -> int:
    """Wrap targeted dense or routed projections in place.

    Runs on the MLX executor right after the loader returns and before every walker
    (patches, offload, fusion). The weights come from mx.load, which maps the file
    lazily, and stay in the dtype the adapter shipped in. The language root is
    model.language_model when present, else model. Each adapter module resolves by its
    path from "layers.<n>." on, as "model.<tail>" or "<tail>" under that root.
    Inkling2 also maps self_attn.o_proj to attn.o_proj when no exact path exists.

    Just before loading, the adapter's config and weights file are checked against
    source.fingerprint, so replaced files cannot be stamped with the cache_revision
    of the files that were discovered. Replace adapter files atomically: a rewrite in
    place after the load is not detected.

    Everything is validated before anything is installed. An AdapterShapeError (an
    unresolved, duplicate or unsupported projection,
    a feature-size mismatch, files changed since discovery, or a model that already
    carries an adapter) therefore leaves the model unmodified. Returns the number of
    wrapped modules.
    """
    from .oqh import runtime as oqh_runtime

    source = binding.source
    if lora_revision_of(model) is not None:
        raise AdapterShapeError(
            "The model already carries a LoRA adapter.", source.path
        )
    language_model = getattr(model, "language_model", None)
    root = model if language_model is None else language_model
    named = dict(root.named_modules())
    _require_unchanged(source)
    weights = mx.load(str(source.weights_path))
    if not isinstance(weights, dict):
        raise AdapterError(
            f"{ADAPTER_WEIGHTS_NAME} did not load as named tensors.", source.path
        )

    plan: dict[tuple[int, str | int], tuple[_Replacement, _LoraProjection]] = {}
    for module in source.modules:
        path = (
            module.runtime_path if isinstance(module, _RoutedModule) else module.hf_path
        )
        name = _target_name(path, named, source.path, getattr(root, "model_type", None))
        parent, key = _locate(root, name)
        slot = (id(parent), key)
        if slot in plan:
            raise AdapterShapeError(
                f"{plan[slot][0].hf_path} and {module.hf_path} both target "
                f"base module {name}.",
                source.path,
            )
        replacement = _plan_replacement(
            parent, key, name, named[name], module, weights, source
        )
        wrapper_type = (
            _LoraSwitchLinear if isinstance(module, _RoutedModule) else LoraLinear
        )
        plan[slot] = (
            replacement,
            wrapper_type(
                replacement.target,
                replacement.lora_a,
                replacement.lora_b,
                source.config.scale,
                binding.cache_revision,
            ),
        )
    _require_unchanged(source)
    installed = []
    try:
        for replacement, wrapper in plan.values():
            _install(replacement.parent, replacement.key, wrapper)
            installed.append(replacement)
    except Exception:
        for replacement in reversed(installed):
            _install(replacement.parent, replacement.key, replacement.target)
        raise
    oqh_runtime.release_wrapped(model)
    return len(plan)


def lora_revision_of(model: Any) -> str | None:
    """The cache_revision stamped on the wrappers under `model`, or None."""
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return None
    for module in modules():
        if isinstance(module, _LoraProjection):
            return module.revision
    return None
