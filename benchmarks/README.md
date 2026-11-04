# Qwen verifier benchmark

`bench_qwen_verify_prework.py` compares upstream GDN prework with oMLX's fused
prework in the real `VLMBatchedEngine` serving path. Native MTP is enabled in
both arms. This does not measure the speedup from enabling MTP.

Example invocation from the repository root:

```sh
uv run --python python3.12 python benchmarks/bench_qwen_verify_prework.py \
	~/.models/scaryrawr/Qwen3.6-35B-A3B-mxfp4 > qwen-result.json
```

The default workload has 512 prompt tokens and 128 generated tokens. Each arm
gets one warmup, followed by three baseline/candidate/candidate/baseline groups.
The model stays loaded throughout. `--pp 2048` selects a longer prompt.
`--fixed-depth 1` selects one draft token instead of the default two.

Fixed draft depth is confined to the benchmark process. Adaptive depth can
change floating-point execution geometry and greedy output tokens, confounding
an exact kernel comparison. Production retains its adaptive policy.

The JSON includes raw trials, median producer-side decode throughput, complete
output token IDs, MTP and fused-kernel call counts, MTP head dtypes, runtime
information, source hashes, and model metadata fingerprints. Weight contents
are not hashed. Logs go to stderr.

The runner fails on early EOS, unequal outputs, cache hits, inactive MTP,
inactive treatment, treatment engagement in the baseline, model-file changes,
or missing telemetry. A failed run is not a performance result. Concurrent GPU
work and thermal drift can still distort timing.

## Measured scope

On an Apple M4 Max with 128 GB unified memory and MLX 0.32.2, the final paired
512-token runs produced these medians:

| Model | Upstream prework | Fused prework | Change |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-mxfp4 | 113.82 tok/s | 118.91 tok/s | +4.47% |
| Qwen3.8-27B-mxfp4 | 40.97 tok/s | 41.27 tok/s | +0.73% |

Each run generated identical tokens across all arms. Earlier repeated runs
measured +3.17% for 35B and +1.19% for 27B. The dense-model gain is small.
A 2048-token 35B run measured 87.48 to 89.06 tok/s, a 1.80% increase.
These measurements cover singleton, fixed-depth-two MTP decode, not general
request latency, prefill, or concurrent throughput.

Both MXFP4 trunks retained BF16 MTP heads on disk and after loading. The 27B
head had 15 tensors, and the 35B head had 20. No model weights were modified.

## Production switch

Fused verifier prework is enabled by default for eligible Qwen3.5-family BF16
blocks with batch size one and sequence lengths 2 through 9.
`OMLX_QWEN35_VERIFY_PREWORK=0` selects upstream prework. Qwen4, ordinary decode,
prefill, masked blocks, and unsupported shapes retain their existing paths.

## Flash Next

`Qwen3.8-Flash-Next-heretic-2-mxfp4` uses `qwen4_exp` and L2 normalization.
The Qwen3.5 verifier fusion above does not apply to this model.

A separate 512-token prompt and 128-token generation run measured 32.57 tok/s
without native MTP. A paired run with the Lightning MTP head resident measured
31.05 tok/s with MTP disabled and 37.66 tok/s with one draft token, a 21.31%
raw throughput increase. The loaded MTP head contained 32 BF16 tensors.

Outputs were stable within each arm but differed between ordinary and MTP
decode. These rates therefore do not establish a token-identical speedup.
They measure existing native MTP, not a new Flash Next optimization.
