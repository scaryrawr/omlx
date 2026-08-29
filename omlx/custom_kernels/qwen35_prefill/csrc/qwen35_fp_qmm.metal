#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

#define define_qwen35_fp_qmm_t(mode, bits)                                     \
  template <typename T, const int BM, const int BK, const int BN>              \
  [[kernel]] void qwen35_##mode##_qmm_t(                                      \
      const device uint32_t* w [[buffer(0)]],                                  \
      const device uint8_t* scales [[buffer(1)]],                              \
      const device T* x [[buffer(2)]],                                         \
      device T* y [[buffer(3)]],                                               \
      const constant int& K [[buffer(4)]],                                     \
      const constant int& N [[buffer(5)]],                                     \
      const constant int& M [[buffer(6)]],                                     \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint lid [[thread_index_in_threadgroup]],                                \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]) {                           \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                           \
                                                                                \
    threadgroup T Xs[BM * BK_padded];                                          \
    threadgroup T Ws[BN * BK_padded];                                          \
                                                                                \
    fp_qmm_t_impl<T, 32, bits, true, BM, BK, BN>(                              \
        w,                                                                     \
        scales,                                                                \
        x,                                                                     \
        y,                                                                     \
        Xs,                                                                    \
        Ws,                                                                    \
        K,                                                                     \
        N,                                                                     \
        M,                                                                     \
        K,                                                                     \
        tid,                                                                   \
        lid,                                                                   \
        simd_gid,                                                              \
        simd_lid);                                                             \
  }

#define instantiate_qwen35_fp_qmm_t(mode, type, bm, bk, bn)                   \
  instantiate_kernel(                                                         \
      "qwen35_" #mode "_qmm_t_" #type "_bm_" #bm "_bk_" #bk "_bn_" #bn,  \
      qwen35_##mode##_qmm_t,                                                   \
      type,                                                                    \
      bm,                                                                      \
      bk,                                                                      \
      bn)

#define instantiate_qwen35_fp_variants(mode)                                  \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 32, 32, 32);                  \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 32, 32, 32);                 \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 32, 64, 32);                  \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 32, 64, 32);                 \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 32, 64, 64);                  \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 32, 64, 64);                 \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 64, 64, 64);                  \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 64, 64, 64);                 \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 16, 64, 64);                  \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 16, 64, 64);                 \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 128, 64, 64);                 \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 128, 64, 64);                \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 128, 64, 32);                 \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 128, 64, 32);                \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 64, 32, 64);                  \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 64, 32, 64);                 \
  instantiate_qwen35_fp_qmm_t(mode, float16_t, 128, 32, 64);                 \
  instantiate_qwen35_fp_qmm_t(mode, bfloat16_t, 128, 32, 64)

define_qwen35_fp_qmm_t(mxfp4, 4);
define_qwen35_fp_qmm_t(mxfp8, 8);

instantiate_qwen35_fp_variants(mxfp4);
instantiate_qwen35_fp_variants(mxfp8);
instantiate_qwen35_fp_qmm_t(mxfp4, bfloat16_t, 32, 64, 16);
