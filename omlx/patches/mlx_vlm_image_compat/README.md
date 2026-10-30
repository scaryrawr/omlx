# mlx-vlm image compatibility

oMLX delegates image model detection, per-model sampling defaults and input
validation to mlx-vlm.

The tokenizer patch forces the native Z-Image tokenizer to load locally with
`trust_remote_code=False`, preserving oMLX's safe model-loading default.

Local generation loading checks supported checkpoint metadata before folder
aliases, matching mlx-vlm's capability probe. The pinned loader otherwise
checks aliases first and can select Qwen for a renamed Mage-Flow checkpoint.
Remote references and checkpoints without a supported metadata match still
use the native loader.
