# mlx-vlm image compatibility

The pinned mlx-vlm version includes native Z-Image generation and img2img from
[PR #1952](https://github.com/Blaizzy/mlx-vlm/pull/1952). oMLX no longer
vendors that model package.

This compatibility layer still vendors the serving-time package from the
unmerged MIT-licensed ERNIE-Image
[PR #1954](https://github.com/Blaizzy/mlx-vlm/pull/1954), commit
`c50b0b363c3fc65b060ad098d4745e44e5c8cfdc`. The upstream conversion CLI is
intentionally excluded. Remove the remaining vendor tree and ERNIE registration
when that PR is included in the mlx-vlm pin.

The small runtime wrappers preserve oMLX's nullable request-default and local
layout dispatch behavior. They also force native Z-Image tokenizer loading to
remain local with `trust_remote_code=False`, preserving oMLX's safe model-loading
default.
