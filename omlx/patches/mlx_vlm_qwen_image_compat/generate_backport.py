# SPDX-License-Identifier: Apache-2.0
"""Generate an exact, in-memory Qwen backport from immutable git revisions."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import pprint
import subprocess
import sys
from pathlib import Path

BASE_REVISION = "4f4634bb813c0298cb1467bed2e957526c71d0b4"
PINNED_REVISION = "36494b42c7676081aa3c14878b3175b90c2fedf4"
PIPELINE_MODULE = "mlx_vlm.models.qwen_image.pipeline"
EAGER_EVAL = "        mx.eval(self.latents_mean, self.latents_std)\n"
MODULES = (
    "mlx_vlm.models.qwen_image.config",
    "mlx_vlm.models.qwen_image.scheduler",
    "mlx_vlm.models.qwen_image.text_encoder",
    PIPELINE_MODULE,
    "mlx_vlm.models.qwen_image.model",
    "mlx_vlm.generate.image",
)
SYMBOLS = {"mlx_vlm.generate.image": "_model_types_from_class_name"}


def selected_source(source: str, symbol: str | None) -> str:
    if symbol is None:
        return source
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == symbol:
            result = ast.get_source_segment(source, node)
            if result is not None:
                return result
    raise ValueError(f"Missing upstream symbol: {symbol}")


def source_hash(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Python 3.12 adds empty type_params fields to non-generic definitions.
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and getattr(node, "type_params", None) == []
        ):
            del node.type_params
    # Python 3.13 omits empty lists by default; retain the older dump format.
    options = {"show_empty": True} if sys.version_info >= (3, 13) else {}
    return hashlib.sha256(ast.dump(tree, **options).encode()).hexdigest()


def replacements(before: str, after: str) -> tuple[tuple[str, str], ...]:
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    edits = []
    for group in difflib.SequenceMatcher(a=old, b=new).get_grouped_opcodes(n=3):
        first, last = group[0], group[-1]
        left = "".join(old[first[1] : last[2]])
        right = "".join(new[first[3] : last[4]])
        if before.count(left) != 1:
            raise ValueError("Backport hunk must have exactly one source match")
        edits.append((left, right))
    return tuple(edits)


def read_sources(repo: Path, revision: str) -> dict[str, str]:
    return {
        name: subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{name.replace('.', '/')}.py"],
            text=True,
        )
        for name in MODULES
    }


def pinned_pipeline_target(base: str, pinned: str, native: str) -> str:
    if pinned.count(EAGER_EVAL) != 1 or pinned.replace(EAGER_EVAL, "", 1) != base:
        raise ValueError("Pinned pipeline differs beyond the expected eager evaluation")
    anchor = pinned[: pinned.index(EAGER_EVAL)].splitlines(keepends=True)[-1]
    if native.count(anchor) != 1:
        raise ValueError("Native pipeline does not have a unique eager evaluation anchor")
    return native.replace(anchor, anchor + EAGER_EVAL, 1)


def render_backport(revision, backports, pinned, changed_hunks) -> str:
    return (
        "# SPDX-License-Identifier: Apache-2.0\n"
        '"""Generated from mlx-vlm; regenerate with generate_backport.py."""\n\n'
        f'SOURCE_REVISION = "{revision}"\n'
        f'BASE_REVISION = "{BASE_REVISION}"\n'
        f'PINNED_REVISION = "{PINNED_REVISION}"\n'
        f"BACKPORTS = {pprint.pformat(backports, width=100, sort_dicts=False)}\n\n"
        f"_PIPELINE_MODULE = {PIPELINE_MODULE!r}\n"
        f"_PINNED_PIPELINE_EDITS = {pprint.pformat(changed_hunks, width=100, sort_dicts=False)}\n"
        "BACKPORT_PROFILES = (\n"
        "    BACKPORTS,\n"
        "    {\n"
        "        **BACKPORTS,\n"
        "        _PIPELINE_MODULE: (\n"
        f"            {pinned[0]!r},\n"
        f"            {pinned[1]!r},\n"
        "            tuple(\n"
        "                _PINNED_PIPELINE_EDITS.get(index, edit)\n"
        "                for index, edit in enumerate(BACKPORTS[_PIPELINE_MODULE][2])\n"
        "            ),\n"
        "        ),\n"
        "    },\n"
        ")\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-revision")
    mode.add_argument("--verify-from-base", action="store_true")
    mode.add_argument("--regenerate-from-base", action="store_true")
    args = parser.parse_args()
    from omlx.patches.mlx_vlm_qwen_image_compat import _prepare_sources

    sources = read_sources(args.source_repo, BASE_REVISION)
    pinned_sources = read_sources(args.source_repo, PINNED_REVISION)
    if args.source_revision is None:
        from omlx.patches.mlx_vlm_qwen_image_compat._backport import (
            BACKPORTS,
            SOURCE_REVISION,
        )

        revision = SOURCE_REVISION
        native_sources = _prepare_sources(sources, (BACKPORTS,))
        if set(native_sources) != set(MODULES):
            raise ValueError("Expected the complete legacy-to-native backport")
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(args.source_repo), "rev-parse", args.source_revision],
            text=True,
        ).strip()
        native_sources = read_sources(args.source_repo, revision)

    backports = {}
    for name in MODULES:
        before = selected_source(sources[name], SYMBOLS.get(name))
        after = selected_source(native_sources[name], SYMBOLS.get(name))
        backports[name] = (
            source_hash(before),
            source_hash(after),
            replacements(before, after),
        )
    if args.source_revision is None and backports != BACKPORTS:
        raise ValueError("Reconstructed target changed the verified legacy backport")

    pinned_native = pinned_pipeline_target(
        sources[PIPELINE_MODULE],
        pinned_sources[PIPELINE_MODULE],
        native_sources[PIPELINE_MODULE],
    )
    pinned = (
        source_hash(pinned_sources[PIPELINE_MODULE]),
        source_hash(pinned_native),
        replacements(pinned_sources[PIPELINE_MODULE], pinned_native),
    )
    legacy_hunks = backports[PIPELINE_MODULE][2]
    if len(legacy_hunks) != len(pinned[2]):
        raise ValueError("Pinned pipeline cannot share the legacy hunk sequence")
    changed_hunks = {
        index: new
        for index, (old, new) in enumerate(zip(legacy_hunks, pinned[2]))
        if old != new
    }
    if not changed_hunks:
        raise ValueError("Pinned pipeline requires a distinct verified recipe")
    profiles = (backports, {**backports, PIPELINE_MODULE: pinned})
    patched = _prepare_sources(pinned_sources, profiles)
    if set(patched) != set(MODULES):
        raise ValueError("Expected the complete pinned-to-native backport")
    if _prepare_sources(native_sources, profiles) or _prepare_sources(patched, profiles):
        raise ValueError("Expected both complete native source profiles to be no-ops")

    output = Path(__file__).with_name("_backport.py")
    rendered = render_backport(revision, backports, pinned, changed_hunks)
    if args.verify_from_base:
        if output.read_text() != rendered:
            raise ValueError("Generated backport does not match the verified sources")
        print(
            "Legacy and pinned backports and generated data verified from local base"
        )
    else:
        output.write_text(rendered)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
