# Serving-path benchmarks

## Qwen verifier benchmark

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

## Inkling 2 loop projections

`OMLX_INKLING2_SPLIT_LOOP_PROJECTIONS=1` enables selective K and V projections
on multi-token looped sliding-attention visits. The switch is read when the
model loads.
It is off by default. Unset it or use `0` to restore the original path.

The original path applies the whole fused projection to three distinct inputs.
The selective path computes only the needed rows for the K and V inputs.
It retains the original weights, loop schedule, per-visit caches, and convolution
states. Shared-input visits and single-token steps use the original projection
path. Dense and affine-quantized linear layers support the selective path.
Other projection types use the original path.
Different matrix shapes can change floating-point rounding and generated tokens.

`bench_qwen_decode.py` supports both `inkling2` and `inkling2_loop`, alongside
Qwen. For Inkling, `--native-mtp on` attaches the bundled DSpark drafter through
the production loader. A missing or failed drafter fails the run. The
`speculative` JSON records attachment, configured depth, and adaptive mode.
Attachment alone does not prove per-request activity. Acceptance counters are
not part of this JSON. The existing MTP logger emits them at INFO.

Run matched baseline and candidate processes with the original checkpoint:

```sh
MODEL="$HOME/.models/thinkingmachines/Inkling-2-Mini"
for pair in 0 1 2 3 4 5; do
	for slot in 0 1; do
		split=$(( (pair + slot) % 2 ))
		OMLX_INKLING2_SPLIT_LOOP_PROJECTIONS="$split" \
			uv run --python python3.12 python benchmarks/bench_qwen_decode.py \
			"$MODEL" --pp 512 --gen 64 --repeats 1 --warmup 1 \
			--native-mtp on --label "pair-$pair-split-$split" \
			> "pair-$pair-split-$split.json"
	done
done
```

The example reverses the order in alternate pairs to reduce drift bias.
Run only one GPU experiment at a time and allow the machine to cool between
pairs. Repeat with `--pp 2048` and `--pp 8192` for longer inputs, and with
`--native-mtp off` for target-only decode. `--mtp-depth N` selects a fixed
Inkling draft depth, capped by the checkpoint's trained depth. Omit it to
retain adaptive depth.

The JSON includes producer-side decode throughput, consumer time to first
token, output IDs, peak MLX memory, and sampled CPU, GPU, memory, and OS thermal
pressure. Time to first token includes prefill and streaming delay. It is not
pure prefill throughput. Missing thermal observations or nonzero pressure fail
the run. Nominal pressure does not establish identical die temperature, power,
or clock speed. Those values are not measured.

Check `status`, every trial's `errors`, exact token counts, output agreement,
and thermal observations before comparing rates. Warmup trials are marked.
Early EOS, cache hits, unstable outputs within an arm, and missing producer
timestamps also fail the run. Stable IDs do not replace held-out quality tests.

### Measured scope

On an M4 Max with 128 GiB memory and MLX 0.32.4, the original
`thinkingmachines/Inkling-2-Mini` checkpoint with adaptive DSpark produced:

| Prompt tokens | Original decode | Selective decode | Original TTFT | Selective TTFT |
| --- | ---: | ---: | ---: | ---: |
| 512 | 62.80 tok/s | 69.00 tok/s | 0.693 s | 0.647 s |
| 2048 | 54.94 tok/s | 53.62 tok/s | 2.644 s | 2.373 s |

Each request generated 64 tokens. The 512-token confirmation used six
independent mirrored process pairs, with one warmup and one measured request
per process. Decode ranges were 62.73 to 62.94 tok/s and 68.91 to 69.06 tok/s.
The paired geometric-mean ratio was 1.098, with a bootstrap 95% interval of
1.097 to 1.100. Peak MLX memory medians were 20.55 and 20.52 GiB.
Outputs matched across both arms. All accepted trials had nominal OS thermal
pressure. GPU utilization averaged 88% to 97% across the short-context trials.

The 2048-token check used two mirrored process pairs and three measured
requests per process. Its decode ranges were 46.49 to 55.96 tok/s and 44.24 to
56.36 tok/s. The wide overlap leaves performance at that length inconclusive.
Within-process repeats slowed by about 16% despite nominal thermal pressure.
Those observations are not independent, and a 2048-token regression remains
unknown rather than ruled out.
Target-only 512-token decode was 45.13 versus 45.36 tok/s, also within noise.
An earlier single-token regression was removed by limiting selective
projections to multi-token steps.

The 8192-token adaptive run failed output stability between warmup and the
measured candidate request, despite nominal thermal pressure. Earlier attempts
also hit thermal pressure. None count as a speedup result. Long-context
stability remains unresolved, so this switch remains experimental and off by
default. The measurements do not reach the 1.5x decode target, cover concurrent
requests, or prove a pure prefill speedup.

Eight complete held-out tasks covered arithmetic, code, JSON, logic, French,
tool invocation, a tool-result continuation, and an image. Across 535
teacher-forced positions, mean KL was 0.000314 and top-1 agreement was 532/535.
Both configurations passed the native-parser answer checks. Seven final answers
matched exactly. The tool-result answers expressed the same facts with different
wording, and some thinking traces differed. These numerical checks cover
prefill followed by single-token steps, not the DSpark verifier's distributions.

Separately, all eight tasks passed through `VLMBatchedEngine.chat` with real
adaptive DSpark on both arms. MTP logs confirmed multi-token verification and
accepted drafts for all 16 requests, including vision and tools. The answers
matched exactly except for equivalent tool-result wording. This covers completed
tasks on the optimized serving path, but remains limited empirical evidence,
not universal quality equivalence.

A fused controller up-projection kernel showed no serving gain and was removed.
An isolated oQ4e conversion was faster in the initial short-context sweep but
failed the preregistered mean-KL threshold of 0.001. It is not a recommended
replacement. Fixed draft depths did not improve the short-context screen.
No oQh support or checkpoint changes are part of this optimization.
