# Qwen3.8 MTP Depth and Head Study

This study tested whether deeper drafting or further MTP-head quantization
could improve `Qwen3.8-27B-oQ4e-mtp` decode on an M4 Max with 128 GB unified
memory. The production defaults remain unchanged: adaptive depth capped at
three, with the MTP fusion projection retained in BF16.

## Depth sweep

Each arm generated 256 greedy tokens from the same prompt with persistent
committed MTP history, prompt priming, compiled singleton MLPs, and fused GDN
decode enabled. The sweep ran in one model residency in the order
3, 5, 8, 8, 5, 3 so later arms were exposed to sustained-load slowdown.

| Maximum depth | Decode samples | Median decode | Draft acceptance |
|---|---|---:|---:|
| 3 | 46.47, 36.52 tok/s | 41.50 tok/s | 292 / 377 (77.5%) |
| 5 | 37.73, 35.61 tok/s | 36.67 tok/s | 292 / 385 (75.8%) |
| 8 | 36.58, 34.24 tok/s | 35.41 tok/s | 307 / 405 (75.8%) |

The adaptive controller rarely selected depths above three. Across the two
depth-8 runs, only 13 draft attempts reached positions four through eight.
Extending the M4 verify kernels from their current six-row limit through nine
rows would therefore optimize a path the controller almost never chooses.

## Fusion projection quantization

The checkpoint's MTP head is already about 314 MB: its decoder attention and
MLP projections are quantized, while `mtp.fc` is a 105 MB BF16
`5120 x 10240` fusion projection. An in-memory Q4/group-64 replacement was
alternated against the original projection for three 256-token runs per arm.

| Fusion projection | Median decode | Aggregate draft acceptance | Median head time |
|---|---:|---:|---:|
| BF16 | 38.11 tok/s | 439 / 572 (76.7%) | 59.55 ms/request |
| Q4/group-64 | 37.84 tok/s | 427 / 555 (76.9%) | 56.48 ms/request |

Q4 saved about 5% of MTP-head time but did not improve end-to-end decode.
Head execution was already under 1% of total measured runtime; the target
verify forward dominated. The small storage reduction and neutral acceptance
do not justify changing the checkpoint or weakening the existing precision
guard.

## Conclusion

For this M4 Max and checkpoint, deeper drafting and additional head
quantization are not productive next steps. Future Qwen3.8 decode work should
target the depth-1-to-3 target verification forward. M5-only NAX/tensor paths
are outside this machine's scope.
