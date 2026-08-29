# Qwen3.5-Family Compiled Decode MLP

oMLX compiles stateless Qwen3.5/3.6/3.8 MLP blocks for singleton decode calls
of up to four tokens. This reduces scheduling overhead and fuses elementwise
work around the quantized matrix multiplications. Prefill, batched decode, and
VLM target-verification calls keep their eager paths.

The optimization is enabled by default. Set
`OMLX_QWEN35_COMPILED_MLP=0` before starting oMLX to disable it.

The route is installed after model loading and other Qwen module transforms,
so compiled traces see final serving weights. Quantized outputs are required
to remain bit-exact to eager execution by focused tests.

## Local policy benchmark

On the development machine, a representative Q4 block with hidden size 1,024
and intermediate size 4,096 measured:

| Shape | Eager | Compiled | Result |
|---|---:|---:|---:|
| Batch 1, one token | 0.1444 ms | 0.1297 ms | 1.114x |
| Batch 4, one token | 0.1805 ms | 0.1891 ms | 0.954x |

The batch-4 regression is why compiled dispatch is restricted to singleton
decode rather than enabled for every small continuous batch.
