# SPDX-License-Identifier: Apache-2.0
"""Regenerate the bounded payload from the reviewed immutable mlx-vlm commit."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "70b9ddc6fad39a7fb7639c9941e2847f3cacd414"
LEGACY = "4f4634bb813c0298cb1467bed2e957526c71d0b4"
UTILS_FUNCTIONS = ("update_module_configs", "load_processor")
EXPERT_SYMBOLS = (
    "GLOBAL_SCALE_SUFFIXES",
    "EXPERT_WEIGHT_SUFFIXES",
    "expert_scale_rows",
    "expand_expert_scales",
    "stack_expert_projection",
    "split_expert_projection",
    "move_expert_projection",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--runtime-revision")
    args = parser.parse_args()
    if args.runtime_revision:
        path = Path(__file__).parent / "source_manifest.json"
        manifest = json.loads(path.read_text())
        paths = set(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(args.source),
                    "ls-tree",
                    "-r",
                    "--name-only",
                    args.runtime_revision,
                ],
                text=True,
            ).splitlines()
        )
        for name, hashes in manifest["guards"].items():
            hashes["upgraded"] = (
                hashlib.sha256(
                    subprocess.check_output(
                        [
                            "git",
                            "-C",
                            str(args.source),
                            "show",
                            f"{args.runtime_revision}:{name}",
                        ]
                    )
                ).hexdigest()
                if name in paths
                else None
            )
        manifest["upgraded"] = args.runtime_revision
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return
    if args.manifest is None:
        parser.error("manifest is required when regenerating the payload")
    inventory_bytes = args.manifest.read_bytes()
    if hashlib.sha256(inventory_bytes).hexdigest() != (
        "c7c8ce1a63ff5311053c6a60fd91793fec4d31ef82c5d1e135a8a6c497147203"
    ):
        raise ValueError("Unreviewed native source manifest")
    inventory = json.loads(inventory_bytes)
    destination = Path(__file__).parent / "vendor"

    def blob(revision, path):
        return subprocess.check_output(
            ["git", "-C", str(args.source), "show", f"{revision}:{path}"]
        )

    files = {}
    roots = ("mlx_vlm.models.minicpmv4_7", "mlx_vlm.speculative.drafters.dspark")
    for row in inventory["runtime_files"]:
        if not any(
            row["module"] == root or row["module"].startswith(root + ".")
            for root in roots
        ):
            continue
        data = blob(REVISION, row["path"])
        if hashlib.sha256(data).hexdigest() != row["final_sha256"]:
            raise ValueError(f"Source hash mismatch: {row['path']}")
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[row["path"]] = row["final_sha256"]

    fragments = {}
    for source_path, symbols, output in (
        ("mlx_vlm/utils.py", UTILS_FUNCTIONS, "loader_functions.py"),
        ("mlx_vlm/models/switch_layers.py", EXPERT_SYMBOLS, "expert_helpers.py"),
    ):
        source = blob(REVISION, source_path).decode()
        statements = []
        found = set()
        for node in ast.parse(source).body:
            names = (
                [node.name]
                if isinstance(node, ast.FunctionDef)
                else (
                    [
                        target.id
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    ]
                    if isinstance(node, ast.Assign)
                    else []
                )
            )
            if any(name in symbols for name in names):
                statements.append(ast.get_source_segment(source, node))
                found.update(names)
        if found != set(symbols):
            raise ValueError(f"Incomplete authoritative fragment: {source_path}")
        data = ("\n\n".join(statements) + "\n").encode()
        (destination / output).write_bytes(data)
        fragments[output] = hashlib.sha256(data).hexdigest()

    source = blob(REVISION, "mlx_vlm/prompt_utils.py").decode()
    formatter = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "MessageFormatter"
    )
    method = next(
        node
        for node in formatter.body
        if isinstance(node, ast.FunctionDef) and node.name == "format_message"
    )
    data = (ast.get_source_segment(source, method) + "\n").encode()
    (destination / "message_formatter.py").write_bytes(data)
    fragments["message_formatter.py"] = hashlib.sha256(data).hexdigest()

    for path in ("LICENSE", "mlx_vlm/models/minicpmv4_7/LICENSE"):
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = blob(REVISION, path)
        target.write_bytes(data)
        files[path] = hashlib.sha256(data).hexdigest()

    verification = {}
    for row in inventory["verification_assets"]:
        path = row["path"]
        if not path.startswith("mlx_vlm/tests/"):
            continue
        data = blob(REVISION, path)
        if hashlib.sha256(data).hexdigest() != row["final_sha256"]:
            raise ValueError(f"Verification source hash mismatch: {path}")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        verification[path] = row["final_sha256"]

    # Guard the complete conservative dependency inventory without installing
    # its unrelated/optional branches. The serving path uses oMLX transport.
    guards = {
        row["path"]: {
            "legacy": row["known_legacy_sha256"],
            "native": row["final_sha256"],
        }
        for row in inventory["runtime_files"]
    }
    (Path(__file__).parent / "source_manifest.json").write_text(
        json.dumps(
            {
                "revision": REVISION,
                "legacy": LEGACY,
                "files": files,
                "fragments": fragments,
                "verification": verification,
                "guards": guards,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
