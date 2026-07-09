# SPDX-License-Identifier: Apache-2.0
"""Helpers for selecting model safetensors shards."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_SHARD_RE = re.compile(
    r"^model-(\d+)-of-(\d+)(?:\.(?:jang|jjqf|mxq))?\.safetensors$"
)
_NON_WEIGHT_SAFETENSORS = {
    "jang_imatrix.safetensors",
    "jangpress-prestacked.safetensors",
    "jangtq_runtime.safetensors",
    "jangtq_stacked.safetensors",
}


def select_safetensors_weight_files(model_path: Path) -> list[Path]:
    """Return safetensors files, choosing one complete numbered shard set.

    Some interrupted downloads can leave multiple concurrent
    ``model-NNNNN-of-MMMMM.safetensors`` sets in the same directory. Legacy
    JANG v1 ``model-NNNNN-of-MMMMM.{jang,jjqf,mxq}.safetensors`` shards use
    the same numbering contract and are treated as model weights. JANG sidecars
    such as calibration imatrices and JANGTQ runtime codebooks are excluded
    before choosing the shard set. When a complete numbered set exists, this
    helper returns only the largest complete set so header scans and size
    estimates do not mix stale and current shards. Non-numbered single-file
    checkpoints are returned unchanged.
    """

    files = sorted(path for path in model_path.glob("*.safetensors") if path.is_file())
    weight_files = [path for path in files if path.name not in _NON_WEIGHT_SAFETENSORS]
    numbered: dict[int, dict[int, Path]] = {}
    for path in weight_files:
        match = _MODEL_SHARD_RE.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        total = int(match.group(2))
        numbered.setdefault(total, {})[index] = path

    if not numbered:
        return weight_files

    complete_sets: list[tuple[int, list[Path]]] = []
    for total, by_index in numbered.items():
        expected = set(range(1, total + 1))
        if set(by_index) == expected:
            complete_sets.append(
                (total, [by_index[index] for index in sorted(by_index)])
            )

    if not complete_sets:
        return weight_files

    total, selected = max(complete_sets, key=lambda item: item[0])
    selected_names = {path.name for path in selected}
    if len(numbered) > 1 or any(
        _MODEL_SHARD_RE.match(path.name) and path.name not in selected_names
        for path in weight_files
    ):
        logger.warning(
            "Multiple safetensors shard sets found in %s; using complete "
            "model-*-of-%05d set (%d file(s)) and ignoring stale/incomplete shards",
            model_path,
            total,
            len(selected),
        )
    return selected
