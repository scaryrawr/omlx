# Qwen3.5-Family Compiled Decode MLP

oMLX compiles stateless Qwen3.5/3.6/3.8 dense MLP blocks for singleton decode calls
of up to four tokens. This reduces scheduling overhead and fuses elementwise
work around the quantized matrix multiplications. Prefill, batched decode, and
VLM target-verification calls keep their eager paths.

The optimization is enabled by default. Set
`OMLX_QWEN35_COMPILED_MLP=0` before starting oMLX to disable it.

The route is installed after model loading and other Qwen module transforms,
so compiled traces see final serving weights. Quantized outputs are required
to remain bit-exact to eager execution by focused tests.

Sparse MoE blocks keep their native routing and shared-expert combination.
Their shared dense MLP still uses compiled decode dispatch. MLX 0.32.2 can
change float32 rounding when whole-block compilation fuses the shared gate's
sigmoid with its multiplication. Keeping that combination outside compilation
preserves exact outputs without changing the routed-decode kernels.

On an Apple M4 Max, a Q4 block with eight experts, hidden size 1,024, and
intermediate size 4,096 took 17-20 microseconds more per one-token call with
this policy than with whole-block compilation. This cost preserves the
bit-exact output contract.

## Local policy benchmark

On the development machine, a representative Q4 block with hidden size 1,024
and intermediate size 4,096 measured:

| Shape | Eager | Compiled | Result |
|---|---:|---:|---:|
| Batch 1, one token | 0.1444 ms | 0.1297 ms | 1.114x |
| Batch 4, one token | 0.1805 ms | 0.1891 ms | 0.954x |

The batch-4 regression is why compiled dispatch is restricted to singleton
decode rather than enabled for every small continuous batch.
