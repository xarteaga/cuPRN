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

#include "sequence.h"
#include "vector.cuh"
#include <chrono>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define SEQUENCE_LENGTH 128

#define BLOCK_SIZE 128
#define NOF_BLOCKS 128

#define USE_SHARED_INPUT 1
#define USE_FIX_POINT 0
#define USE_REDUCTION 1

__global__ void
cuprn_find(const cuprn_context_t* context, const float* input, cuprn_result_t* result, uint32_t length, uint32_t seed_N)
{

  __shared__ float    corr[BLOCK_SIZE];
  __shared__ uint32_t seeds[BLOCK_SIZE];

  uint32_t thread_id_block    = threadIdx.x;
  uint32_t block_id           = blockIdx.x;
  uint32_t thread_id_grid     = blockDim.x * block_id + thread_id_block;
  uint32_t nof_seeds          = 1U << seed_N;
  uint32_t nof_threads        = (uint32_t)gridDim.x * (uint32_t)blockDim.x;
  uint32_t nof_seeds_x_thread = nof_seeds / nof_threads;

#if USE_SHARED_INPUT
  // Trim input to the maximum sequence length that fits in the shared memory
  if (length > BLOCK_SIZE) {
    length = BLOCK_SIZE;
  }

#if USE_FIX_POINT
  // Load shared memory
  __shared__ int16_t shared_input[SEQUENCE_LENGTH];
  uint32_t           nof_elements_thread = (length + blockDim.x - 1) / blockDim.x;
  for (uint32_t i = 0; i < nof_elements_thread; i++) {
    uint32_t idx = nof_elements_thread * threadIdx.x + i;
    if (idx < length) {
      shared_input[idx] = 1024 * input[idx];
    }
  }
  __syncthreads();
#else

  // Load shared memory
  __shared__ float shared_input[SEQUENCE_LENGTH];
  uint32_t         nof_elements_thread = (length + blockDim.x) / blockDim.x;
  for (uint32_t i = 0; i < nof_elements_thread; i++) {
    uint32_t idx = nof_elements_thread * threadIdx.x + i;
    if (idx < length) {
      shared_input[idx] = input[idx];
    }
  }
  __syncthreads();
#endif
#endif

  // Calculate thread seed boundaries
  uint32_t seed_begin = nof_seeds_x_thread * thread_id_grid;
  uint32_t seed_end   = nof_seeds_x_thread * (thread_id_grid + 1);

  float    max_corr = 0.0f;
  uint32_t max_seed = 0;

  // Find maximum correlation and seed for the thread boundaries
  for (uint32_t seed = seed_begin; seed < seed_end; seed++) {
#if USE_SHARED_INPUT
#if USE_FIX_POINT
    float temp = cuprn_sequence_correlate_device_s(context, shared_input, length, seed);
#else
    float temp = cuprn_sequence_correlate_device_f(context, shared_input, length, seed);
#endif
#else
    float temp = cuprn_sequence_correlate_device_f(context, input, length, seed);
#endif
    if (temp > max_corr) {
      max_seed = seed;
      max_corr = temp;
    }
  }

  corr[thread_id_block]  = max_corr;
  seeds[thread_id_block] = max_seed;

#if USE_REDUCTION
  // Reduction algorithm in the block
  for (uint32_t N = blockDim.x / 2; N != 0; N /= 2) {
    __syncthreads();

    if (threadIdx.x < N) {
      if (corr[threadIdx.x + N] > corr[threadIdx.x]) {
        corr[threadIdx.x]  = corr[threadIdx.x + N];
        seeds[threadIdx.x] = seeds[threadIdx.x + N];
      }
    }
  }
#else
  __syncthreads();
  if (thread_id_block == 0) {
    for (uint32_t i = 0; i < blockDim.x; i++) {
      if (corr[i] > corr[0]) {
        seeds[0] = seeds[i];
        corr[0] = corr[i];
      }
    }
  }
#endif

  if (thread_id_block == 0) {
    result[block_id].correlation        = corr[0];
    result[block_id].seed               = seeds[0];
    result[block_id].thread_id          = thread_id_grid;
    result[block_id].block_id           = block_id;
    result[block_id].nof_seeds_x_thread = nof_seeds_x_thread;
    result[block_id].nof_threads        = nof_threads;
  }
}

int main()
{
  uint32_t         seed_gold      = 0x12345678;
  uint32_t         seed_N         = 31;
  float*           x_device       = cuprn_device_malloc_f(SEQUENCE_LENGTH);
  float*           x_host         = cuprn_host_malloc_f(SEQUENCE_LENGTH);
  cuprn_context_t* context_device = cuprn_context_init_device();
  cuprn_result_t*  result_device  = cuprn_result_init_device(NOF_BLOCKS);
  cuprn_result_t*  result_host    = cuprn_result_init_host(NOF_BLOCKS);

  // Check allocation is OK
  if (x_device && x_host && result_device && result_host && context_device) {
    // Generate sequence
    cuprn_sequence_gen_device_f<<<1, 1>>>(context_device, 1, x_device, SEQUENCE_LENGTH, seed_gold);

    // Print the simulation data
    printf("Seed: 0x%08x; Seed size: %d; Sequence length: %d; Nof CUDA blocks: %d",
           seed_gold,
           seed_N,
           SEQUENCE_LENGTH,
           NOF_BLOCKS);

    // Brute force sequence to find seed
    auto start = std::chrono::system_clock::now();
    cuprn_find<<<NOF_BLOCKS, BLOCK_SIZE>>>(context_device, x_device, result_device, SEQUENCE_LENGTH, seed_N);

    cuprn_result_device_2_host(result_host, result_device, NOF_BLOCKS);
    auto end = std::chrono::system_clock::now();

    cuprn_copy_device_2_host_f(x_host, x_device, SEQUENCE_LENGTH);

    // Print original sequence if uncommented
    //    for (uint32_t i = 0; i < SEQUENCE_LENGTH; i++) {
    //      printf("x[%d]=%+.3f\n", i, x_host[i]);
    //    }

    // Find best data in block
    float    max_corr   = 0;
    uint32_t best_block = 0;
    for (uint32_t i = 0; i < NOF_BLOCKS; i++) {
      // Print Blocks data
      printf("-- seed=%08x; corr=%f; thread_id=%d; block_id=%d; nof_seeds_x_thread=%d; nof_threads=%d;\n",
             result_host[i].seed,
             result_host[i].correlation,
             result_host[i].thread_id,
             result_host[i].block_id,
             result_host[i].nof_seeds_x_thread,
             result_host[i].nof_threads);
      if (result_host[i].correlation > max_corr) {
        best_block = i;
        max_corr   = result_host[i].correlation;
      }
    }

    // Print best block
    printf(
        "seed=%08x; match=%s; corr=%f; thread_id=%d; block_id=%d; nof_seeds_x_thread=%d; nof_threads=%d; duration=%ld "
        "milliseconds\n",
        result_host[best_block].seed,
        seed_gold == result_host[best_block].seed ? "yes" : "no",
        result_host[best_block].correlation,
        result_host[best_block].thread_id,
        result_host[best_block].block_id,
        result_host[best_block].nof_seeds_x_thread,
        result_host[best_block].nof_threads,
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
  }

  cuprn_device_free_f(x_device);
  cuprn_host_free_f(x_host);
  cuprn_context_device_free(context_device);
  cuprn_result_device_free(result_device);
  cuprn_result_host_free(result_host);
  return 0;
}
