# SPDX-License-Identifier: Apache-2.0
"""Portable source fingerprint checks without MLX or model dependencies."""

import ast

import pytest

from omlx.patches.mlx_vlm_qwen_image_compat.generate_backport import source_hash


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "class Model:\n    pass\n",
            "982ac2b3eb77fb75af373b2638159f4a698e3701e533a1c4d69bfe2f582f03f3",
        ),
        (
            "def load():\n    return 1\n",
            "577391320d95bead4088cf34bd51c4f9c18b81d1572ad1ae17b997a4a861e95b",
        ),
        (
            "async def load():\n    return 1\n",
            "cd3a2dc5966095d9d58947f47478e725957eb3c754cbe75254cb654307677f92",
        ),
    ],
)
def test_source_fingerprints_are_python_version_independent(
    source, expected, monkeypatch
):
    assert source_hash(source) == expected
    tree = ast.parse(source)
    definition = tree.body[0]
    if "type_params" not in definition._fields:
        definition._fields = (*definition._fields, "type_params")
    definition.type_params = []
    monkeypatch.setattr(ast, "parse", lambda _: tree)
    assert source_hash(source) == expected
    definition.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert source_hash(source) != expected


@pytest.mark.parametrize(
    ("source", "changed", "expected"),
    [
        (
            "values = []\n",
            "values = [1]\n",
            "ab546b0086b83a8aaacb0a6c9033493ed1eb9e07bf8012b77cf9e5be261f2f69",
        ),
        (
            "load = lambda: 1\n",
            "load = lambda x: 1\n",
            "0c5ebbca9fd07eb4389ad437f1cd11c603f0f5c332c264c35f8af5528a87bb41",
        ),
        (
            "load()\n",
            "load(1)\n",
            "2448683c81365c813b6bdfb70e67090fac996d078ebdf5c9a2fb6317870b1243",
        ),
    ],
)
def test_source_fingerprints_preserve_empty_fields(source, changed, expected):
    assert source_hash(source) == expected
    assert source_hash(changed) != expected
