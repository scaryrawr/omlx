# SPDX-License-Identifier: Apache-2.0
"""Generate an exact, in-memory Qwen backport from immutable git revisions."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import pprint
import subprocess
from pathlib import Path

BASE_REVISION = "4f4634bb813c0298cb1467bed2e957526c71d0b4"
MODULES = (
    "mlx_vlm.models.qwen_image.config",
    "mlx_vlm.models.qwen_image.scheduler",
    "mlx_vlm.models.qwen_image.text_encoder",
    "mlx_vlm.models.qwen_image.pipeline",
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
    return hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_repo), "rev-parse", args.source_revision],
        text=True,
    ).strip()
    backports = {}
    for name in MODULES:
        path = name.replace(".", "/") + ".py"
        sources = [
            subprocess.check_output(
                ["git", "-C", str(args.source_repo), "show", f"{ref}:{path}"],
                text=True,
            )
            for ref in (BASE_REVISION, revision)
        ]
        before, after = (
            selected_source(source, SYMBOLS.get(name)) for source in sources
        )
        backports[name] = (
            source_hash(before),
            source_hash(after),
            replacements(before, after),
        )
    output = Path(__file__).with_name("_backport.py")
    output.write_text(
        "# SPDX-License-Identifier: Apache-2.0\n"
        '"""Generated from mlx-vlm; regenerate with generate_backport.py."""\n\n'
        f'SOURCE_REVISION = "{revision}"\n'
        f'BASE_REVISION = "{BASE_REVISION}"\n'
        f"BACKPORTS = {pprint.pformat(backports, width=100, sort_dicts=False)}\n"
    )


if __name__ == "__main__":
    main()
