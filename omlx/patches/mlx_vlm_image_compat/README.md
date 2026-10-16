# mlx-vlm image compatibility

The pinned mlx-vlm version includes native Z-Image and ERNIE-Image generation
and img2img support. oMLX no longer vendors either model package.

The small runtime wrappers preserve oMLX's nullable request-default behavior.
They also force native Z-Image tokenizer loading to remain local with
`trust_remote_code=False`, preserving oMLX's safe model-loading default.
