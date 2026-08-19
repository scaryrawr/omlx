# mlx-vlm image compatibility

This patch vendors the runtime model packages from two unmerged MIT-licensed
mlx-vlm pull requests against upstream main commit
`20eec6cb5564c6a196b046d869d2081c29e3ff92`:

- Z-Image generation and img2img: PR
  [#1952](https://github.com/Blaizzy/mlx-vlm/pull/1952), commit
  `4301adf69d523fcf85095c9ac9c1bbdeab6b3754`.
- ERNIE-Image generation and img2img: PR
  [#1954](https://github.com/Blaizzy/mlx-vlm/pull/1954), commit
  `c50b0b363c3fc65b060ad098d4745e44e5c8cfdc`.

Only serving-time packages are included. The upstream conversion CLIs are
intentionally excluded. Once both pull requests are available at oMLX's
mlx-vlm pin, remove the vendor tree and the registration/backport module after
confirming the upstream generic image APIs retain the nullable request-default
and local-layout dispatch behavior covered by `tests/test_mlx_vlm_image_compat.py`.

The Z-Image tokenizer loader is the sole oMLX source delta: it keeps
`local_files_only=True` but disables remote-code execution to preserve oMLX's
safe model-loading default.
