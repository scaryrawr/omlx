# SPDX-License-Identifier: Apache-2.0
"""Source metadata admission is separate from shape/tokenizer compatibility."""

import hashlib
import json

import pytest

from omlx.speculative.minicpm_pairing import HELPER, TARGET, validate_model_pair


def source(directory, identity, *, conversion=False):
    directory.mkdir()
    config = {"model_type": "minicpmv4_7"}
    if conversion:
        config["omlx_source"] = {"repo_id": identity[0], "revision": identity[1]}
    data = json.dumps(config).encode()
    (directory / "config.json").write_bytes(data)
    if not conversion:
        metadata = directory / ".cache/huggingface/download"
        metadata.mkdir(parents=True)
        blob = b"blob " + str(len(data)).encode() + b"\0" + data
        digest = hashlib.sha1(blob, usedforsecurity=False).hexdigest()
        (metadata / "config.json.metadata").write_text(f"{identity[1]}\n{digest}\n0\n")
    return str(directory)


@pytest.mark.parametrize("conversion", [False, True])
def test_original_and_declared_conversion_pairing(tmp_path, conversion):
    target = source(tmp_path / "target", TARGET, conversion=conversion)
    helper = source(tmp_path / "helper", HELPER, conversion=conversion)
    validate_model_pair(target, helper)


def test_missing_or_stale_provenance_is_not_shape_based_admission(tmp_path):
    target = source(tmp_path / "target", TARGET)
    helper = source(tmp_path / "helper", HELPER)
    with pytest.raises(ValueError, match="source path"):
        validate_model_pair(None, helper)
    (tmp_path / "helper/config.json").write_text('{"model_type": "qwen3"}')
    with pytest.raises(ValueError, match="Stale or mismatched"):
        validate_model_pair(target, helper)
    (tmp_path / "helper/.cache/huggingface/download/config.json.metadata").unlink()
    with pytest.raises(ValueError, match="Cannot establish"):
        validate_model_pair(target, helper)


def test_other_source_or_helper_revision_is_rejected(tmp_path):
    target = source(tmp_path / "target", TARGET, conversion=True)
    helper = source(tmp_path / "helper", (HELPER[0], "unreviewed"), conversion=True)
    with pytest.raises(ValueError, match="does not match"):
        validate_model_pair(target, helper)
