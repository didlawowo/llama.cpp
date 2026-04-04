/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Header declarations for ggml-turbo-quant.c
 */

#pragma once

#include "ggml-common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
