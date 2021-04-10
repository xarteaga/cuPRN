/**
 *
 * \section COPYRIGHT
 *
 * Copyright 2020-2021 Xavier Arteaga
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

#ifndef SRSLTE_SEQUENCE_H
#define SRSLTE_SEQUENCE_H

#include <stdint.h>

#define SEQUENCE_SEED_LEN (31)

typedef struct {
  uint32_t x1_init;
  uint32_t x2_init[SEQUENCE_SEED_LEN];
} cuprn_context_t;

cuprn_context_t* cuprn_context_init_device();

cuprn_context_t* cuprn_context_init_host();

void cuprn_sequence_gen_host_f(cuprn_context_t* context, float amplitude, float* out, uint32_t length, uint32_t seed);

__global__ void
cuprn_sequence_gen_device_f(cuprn_context_t* context, float amplitude, float* out, uint32_t length, uint32_t seed);

__device__ float
cuprn_sequence_correlate_device_f(const cuprn_context_t* context, const float* input, uint32_t length, uint32_t seed);

__device__ float
cuprn_sequence_correlate_device_s(const cuprn_context_t* context, const int16_t* input, uint32_t length, uint32_t seed);

void cuprn_context_device_free(cuprn_context_t* context);

void cuprn_context_host_free(cuprn_context_t* context);

typedef struct {
  uint32_t seed;
  float    correlation;
  uint32_t thread_id;
  uint32_t block_id;
  uint32_t nof_seeds_x_thread;
  uint32_t nof_threads;
} cuprn_result_t;

cuprn_result_t* cuprn_result_init_device(uint32_t nof_blocks);

cuprn_result_t* cuprn_result_init_host(uint32_t nof_blocks);

void cuprn_result_device_free(cuprn_result_t* res);

void cuprn_result_host_free(cuprn_result_t* res);

void cuprn_result_device_2_host(cuprn_result_t* dst, const cuprn_result_t* src, uint32_t nof_blocks);

#endif // SRSLTE_SEQUENCE_H
