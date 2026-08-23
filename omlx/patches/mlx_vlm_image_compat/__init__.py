# SPDX-License-Identifier: Apache-2.0
"""Image-model compatibility layer for the pinned mlx-vlm dependency.

Z-Image is native in the pinned mlx-vlm version. This layer retains the
serving-time portion of unmerged PR #1954 (ERNIE-Image, synchronized with
the scaryrawr-expert-carnival fork at
f991a54bdaf720d3c8e822f3f3893410ff301f6b), plus oMLX's generic nullable
request defaults and safe Z-Image tokenizer policy.

The vendored ERNIE-Image package is appended to the real ``mlx_vlm.models``
namespace so an upstream package takes precedence once the PR merges.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def apply_mlx_vlm_image_compat_patch() -> bool:
    """Register vendored ERNIE-Image and oMLX's image API compatibility."""
    global _APPLIED
    if _APPLIED:
        return False

    import mlx_vlm
    import mlx_vlm.models

    _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")

    image = importlib.import_module("mlx_vlm.generate.image")
    edit_image = importlib.import_module("mlx_vlm.generate.edit_image")
    _patch_request_helpers(image, edit_image)
    _patch_model_type_resolution(image, edit_image)
    _patch_default_resolution(image, edit_image)

    # Importing the adapters proves they can be dispatched through the same
    # generic APIs as upstream image families without loading model weights.
    importlib.import_module("mlx_vlm.models.z_image")
    importlib.import_module("mlx_vlm.models.ernie_image")
    _patch_z_image_tokenizer_security()
    _patch_upstream_family_defaults()
    _clear_model_class_caches(image, edit_image)

    _APPLIED = True
    logger.info("mlx-vlm image compatibility patch applied")
    return True


def is_applied() -> bool:
    """Return whether the compatibility layer has been registered."""
    return _APPLIED


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.append(path_str)


def _patch_request_helpers(image: Any, edit_image: Any) -> None:
    generation_request = image.ImageGenerationRequest
    _patch_nullable_request_defaults(
        generation_request,
        ("steps", "width", "height", "guidance"),
    )
    if not hasattr(generation_request, "resolve_steps"):

        def resolve_steps(self, default: int = image.DEFAULT_IMAGE_STEPS) -> int:
            return default if self.steps is None else self.steps

        generation_request.resolve_steps = resolve_steps

    if not hasattr(generation_request, "resolve_guidance"):

        def resolve_guidance(
            self, default: float = image.DEFAULT_IMAGE_GUIDANCE
        ) -> float:
            return default if self.guidance is None else self.guidance

        generation_request.resolve_guidance = resolve_guidance

    edit_request = edit_image.ImageEditRequest
    _patch_nullable_request_defaults(edit_request, ("steps", "guidance"))
    if not hasattr(edit_request, "resolve_steps"):

        def resolve_steps(self, default: int = image.DEFAULT_IMAGE_STEPS) -> int:
            return default if self.steps is None else self.steps

        edit_request.resolve_steps = resolve_steps

    if not hasattr(edit_request, "resolve_guidance"):

        def resolve_guidance(
            self, default: float = image.DEFAULT_IMAGE_GUIDANCE
        ) -> float:
            return default if self.guidance is None else self.guidance

        edit_request.resolve_guidance = resolve_guidance


def _patch_nullable_request_defaults(
    request_class: Any, fields: tuple[str, ...]
) -> None:
    """Backport ``None`` defaults without replacing mlx-vlm dataclasses."""
    if getattr(request_class, "_omlx_nullable_image_defaults", False):
        return
    original_init = request_class.__init__
    field_order = tuple(request_class.__dataclass_fields__)
    field_positions = {name: field_order.index(name) for name in fields}

    def init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        for name, position in field_positions.items():
            if name not in kwargs and len(args) <= position:
                setattr(self, name, None)

    init._omlx_nullable_image_defaults = True
    init._omlx_original = original_init
    request_class.__init__ = init
    request_class._omlx_nullable_image_defaults = True


def _patch_model_type_resolution(image: Any, edit_image: Any) -> None:
    original_type_from_id = image._model_type_from_id
    if not getattr(original_type_from_id, "_omlx_image_compat", False):

        def model_type_from_id(model: str) -> str:
            model_type = original_type_from_id(model)
            return {
                "z": "z_image",
                "zimage": "z_image",
                "ernie": "ernie_image",
            }.get(model_type, model_type)

        model_type_from_id._omlx_image_compat = True
        model_type_from_id._omlx_original = original_type_from_id
        image._model_type_from_id = model_type_from_id
        # edit_image imports this helper by value, so update its binding too.
        edit_image._model_type_from_id = model_type_from_id

    original_from_indexes = image._image_model_type_from_component_indexes
    if getattr(original_from_indexes, "_omlx_image_compat", False):
        return

    def model_type_from_component_indexes(root: Path) -> str | None:
        model_type = original_from_indexes(root)
        if model_type is not None:
            return model_type
        metadata = image._load_json_file(
            Path(root) / "transformer" / "model.safetensors.index.json"
        )
        weight_map = metadata.get("weight_map") if metadata is not None else None
        if not isinstance(weight_map, dict):
            return None
        keys = set(weight_map)
        if {
            "layers.0.feed_forward.w1.weight",
            "context_refiner.0.attention.to_q.weight",
            "noise_refiner.0.adaLN_modulation.0.weight",
        } <= keys:
            return "z_image"
        if {
            "adaln_modulation.weight",
            "final_norm.linear.weight",
            "layers.0.adaLN_sa_ln.weight",
        } <= keys:
            return "ernie_image"
        return None

    model_type_from_component_indexes._omlx_image_compat = True
    model_type_from_component_indexes._omlx_original = original_from_indexes
    image._image_model_type_from_component_indexes = model_type_from_component_indexes


def _patch_default_resolution(image: Any, edit_image: Any) -> None:
    original_generate_image = image.generate_image
    if not getattr(original_generate_image, "_omlx_image_compat", False):

        def generate_image(
            model: Any,
            request: Any,
            *,
            task: str = image.DEFAULT_IMAGE_TASK,
            image_paths: Any = None,
            output_path: Any = None,
            **kwargs: Any,
        ) -> Any:
            if task == "generate" and isinstance(request, image.ImageGenerationRequest):
                request = _resolve_generation_request(image, model, request)
            return original_generate_image(
                model,
                request,
                task=task,
                image_paths=image_paths,
                output_path=output_path,
                **kwargs,
            )

        generate_image._omlx_image_compat = True
        generate_image._omlx_original = original_generate_image
        image.generate_image = generate_image

    original_edit_image = edit_image.edit_image
    if getattr(original_edit_image, "_omlx_image_compat", False):
        return

    def edit_image_with_defaults(
        model: Any,
        request: Any,
        *,
        image_paths: Any = None,
        output_path: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(request, edit_image.ImageEditRequest):
            request = _resolve_edit_request(image, model, request)
        return original_edit_image(
            model,
            request,
            image_paths=image_paths,
            output_path=output_path,
            **kwargs,
        )

    edit_image_with_defaults._omlx_image_compat = True
    edit_image_with_defaults._omlx_original = original_edit_image
    edit_image.edit_image = edit_image_with_defaults


def _patch_z_image_tokenizer_security() -> None:
    """Keep native Z-Image tokenizer loading local and remote-code-free."""
    pipeline = importlib.import_module("mlx_vlm.models.z_image.pipeline")
    tokenizer = pipeline.AutoTokenizer
    if getattr(tokenizer, "_omlx_safe_z_image_tokenizer", False):
        return

    class SafeAutoTokenizer:
        _omlx_safe_z_image_tokenizer = True

        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            kwargs["local_files_only"] = True
            kwargs["trust_remote_code"] = False
            return tokenizer.from_pretrained(*args, **kwargs)

    pipeline.AutoTokenizer = SafeAutoTokenizer


def _patch_upstream_family_defaults() -> None:
    """Install model defaults used by oMLX's nullable request handling."""
    flux2 = importlib.import_module("mlx_vlm.models.flux2.model")
    mage_flow = importlib.import_module("mlx_vlm.models.mage_flow.model")
    for class_name in ("Flux2ImageGenerationModel", "Flux2ImageEditModel"):
        model_class = getattr(flux2, class_name)
        _install_default_property(model_class, "default_steps", lambda self: 4)
        _install_default_property(model_class, "default_guidance", lambda self: 1.0)
    for class_name in ("MageFlowImageGenerationModel", "MageFlowImageEditModel"):
        model_class = getattr(mage_flow, class_name)
        _install_default_property(
            model_class,
            "default_steps",
            lambda self: self.pipeline.variant.default_steps,
        )
        _install_default_property(
            model_class,
            "default_guidance",
            lambda self: self.pipeline.variant.default_guidance,
        )

    z_image = importlib.import_module("mlx_vlm.models.z_image.model")
    generation = z_image.ZImageGenerationModel
    _install_default_property(
        generation,
        "default_steps",
        lambda self: self.pipeline.config.default_steps,
    )
    _install_default_property(
        generation,
        "default_guidance",
        lambda self: self.pipeline.config.default_guidance,
    )
    _install_default_property(generation, "default_width", lambda self: 1024)
    _install_default_property(generation, "default_height", lambda self: 1024)

    edit = z_image.ZImageEditModel
    _install_default_property(
        edit,
        "default_steps",
        lambda self: 8 if self.variant == "turbo" else 50,
    )
    _install_default_property(
        edit,
        "default_guidance",
        lambda self: self.pipeline.config.default_guidance,
    )


def _install_default_property(model_class: Any, name: str, getter: Any) -> None:
    if not hasattr(model_class, name):
        setattr(model_class, name, property(getter))


def _model_default(model: Any, name: str, fallback: int | float) -> int | float:
    value = getattr(model, name, None)
    return fallback if value is None else value


def _resolve_generation_request(image: Any, model: Any, request: Any) -> Any:
    updates: dict[str, int | float] = {}
    if request.width is None:
        updates["width"] = int(_model_default(model, "default_width", 512))
    if request.height is None:
        updates["height"] = int(_model_default(model, "default_height", 512))
    if request.steps is None:
        updates["steps"] = int(
            _model_default(model, "default_steps", image.DEFAULT_IMAGE_STEPS)
        )
    if request.guidance is None:
        updates["guidance"] = float(
            _model_default(model, "default_guidance", image.DEFAULT_IMAGE_GUIDANCE)
        )
    return replace(request, **updates) if updates else request


def _resolve_edit_request(image: Any, model: Any, request: Any) -> Any:
    updates: dict[str, int | float] = {}
    if request.steps is None:
        updates["steps"] = int(
            _model_default(model, "default_steps", image.DEFAULT_IMAGE_STEPS)
        )
    if request.guidance is None:
        updates["guidance"] = float(
            _model_default(model, "default_guidance", image.DEFAULT_IMAGE_GUIDANCE)
        )
    return replace(request, **updates) if updates else request


def _clear_model_class_caches(image: Any, edit_image: Any) -> None:
    for name in ("_image_model_class_for_type",):
        cache = getattr(image, name, None)
        clear = getattr(cache, "cache_clear", None)
        if callable(clear):
            clear()
    for name in ("_image_edit_model_class_for_type",):
        cache = getattr(edit_image, name, None)
        clear = getattr(cache, "cache_clear", None)
        if callable(clear):
            clear()


__all__ = ["apply_mlx_vlm_image_compat_patch", "is_applied"]
