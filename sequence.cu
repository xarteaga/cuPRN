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

#include "memory.cuh"
#include "sequence.h"

CUPRN_MEMORY(cuprn_context_t, ctx)

CUPRN_MEMORY(cuprn_result_t, res)

/**
 * Nc parameter defined in 3GPP. Do not change.
 */
#define SEQUENCE_NC (1600)

/*
 * Pseudo Random Sequence generation.
 * It follows the 3GPP Release 8 (LTE) 36.211
 * Section 7.2
 */

/**
 * Parallel bit generation for x1/x2 sequences parameters. Exploits the fact that the sequence generation is 31 chips
 * ahead and the maximum register shift is 3 (for x2).
 */
#define SEQUENCE_PAR_BITS (28U)
#define SEQUENCE_MASK ((1U << SEQUENCE_PAR_BITS) - 1U)

/**
 * Computes one step of the X1 sequence for SEQUENCE_PAR_BITS simultaneously
 * @param state 32 bit current state
 * @return new 32 bit state
 */
static inline uint32_t sequence_gen_LTE_pr_memless_step_par_x1(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 3U);

    // Prepare feedback
    f = ((f & SEQUENCE_MASK) << (SEQUENCE_SEED_LEN - SEQUENCE_PAR_BITS));

    // Insert feedback
    state = (state >> SEQUENCE_PAR_BITS) ^ f;

    return state;
}

__device__ uint32_t cuprn_step_x1_par(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 3U);

    // Prepare feedback
    f = ((f & SEQUENCE_MASK) << (SEQUENCE_SEED_LEN - SEQUENCE_PAR_BITS));

    // Insert feedback
    state = (state >> SEQUENCE_PAR_BITS) ^ f;

    return state;
}

/**
 * Computes one step of the X1 sequence for 1bit
 * @param state 32 bit current state
 * @return new 32 bit state
 */
static inline uint32_t sequence_gen_LTE_pr_memless_step_x1(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 3U);

    // Prepare feedback
    f = ((f & 1U) << (SEQUENCE_SEED_LEN - 1U));

    // Insert feedback
    state = (state >> 1U) ^ f;

    return state;
}

__device__ uint32_t cuprn_step_x1(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 3U);

    // Prepare feedback
    f = ((f & 1U) << (SEQUENCE_SEED_LEN - 1U));

    // Insert feedback
    state = (state >> 1U) ^ f;

    return state;
}

__device__ uint32_t cuprn_step_x2(uint32_t state) {
// Perform XOR
    uint32_t f = state ^(state >> 1U) ^(state >> 2U) ^(state >> 3U);

    // Prepare feedback
    f = ((f & 1U) << (SEQUENCE_SEED_LEN - 1U));

    // Insert feedback
    state = (state >> 1U) ^ f;

    return state;
}

/**
 * Computes one step of the X2 sequence for SEQUENCE_PAR_BITS simultaneously
 * @param state 32 bit current state
 * @return new 32 bit state
 */
static inline uint32_t sequence_gen_LTE_pr_memless_step_par_x2(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 1U) ^(state >> 2U) ^(state >> 3U);

    // Prepare feedback
    f = ((f & SEQUENCE_MASK) << (SEQUENCE_SEED_LEN - SEQUENCE_PAR_BITS));

    // Insert feedback
    state = (state >> SEQUENCE_PAR_BITS) ^ f;

    return state;
}

__device__ uint32_t cuprn_step_x2_par(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 1U) ^(state >> 2U) ^(state >> 3U);

    // Prepare feedback
    f = ((f & SEQUENCE_MASK) << (SEQUENCE_SEED_LEN - SEQUENCE_PAR_BITS));

    // Insert feedback
    state = (state >> SEQUENCE_PAR_BITS) ^ f;

    return state;
}

/**
 * Computes one step of the X2 sequence for 1bit
 * @param state 32 bit current state
 * @return new 32 bit state
 */
static inline uint32_t sequence_gen_LTE_pr_memless_step_x2(uint32_t state) {
    // Perform XOR
    uint32_t f = state ^(state >> 1U) ^(state >> 2U) ^(state >> 3U);

    // Prepare feedback
    f = ((f & 1U) << (SEQUENCE_SEED_LEN - 1U));

    // Insert feedback
    state = (state >> 1U) ^ f;

    return state;
}


static inline void cuprn_context_init(cuprn_context_t *context) {

    // Compute transition step
    context->x1_init = 1;
    for (uint32_t n = 0; n < SEQUENCE_NC; n++) {
        context->x1_init = sequence_gen_LTE_pr_memless_step_x1(context->x1_init);
    }

    // For each bit of the seed
    for (uint32_t i = 0; i < SEQUENCE_SEED_LEN; i++) {
        // Compute transition step
        context->x2_init[i] = 1U << i;
        for (uint32_t n = 0; n < SEQUENCE_NC; n++) {
            context->x2_init[i] = sequence_gen_LTE_pr_memless_step_x2(context->x2_init[i]);
        }
    }
}

cuprn_context_t *cuprn_context_init_host() {
    cuprn_context_t *context = cuprn_host_malloc_ctx(1);
    if (context != nullptr) {
        cuprn_context_init(context);
    }
    return context;
}

cuprn_context_t *cuprn_context_init_device() {
    cuprn_context_t context = {};
    cuprn_context_init(&context);

    cuprn_context_t *context2 = cuprn_device_malloc_ctx(1);
    if (context2 != nullptr) {
        cuprn_copy_host_2_device_ctx(context2, &context, 1);
    }
    return context2;
}

void cuprn_context_device_free(cuprn_context_t *context) {
    cuprn_device_free_ctx(context);
}

void cuprn_context_host_free(cuprn_context_t *context) {
    cuprn_device_free_ctx(context);
}


__device__ uint32_t cuprn_x1_init(const cuprn_context_t *context) {
    return context->x1_init;
}

__device__ uint32_t cuprn_x2_init(const cuprn_context_t *context, uint32_t seed) {
    uint32_t x2 = 0;

    for (uint32_t i = 0; i < SEQUENCE_SEED_LEN; i++) {
        if ((seed >> i) & 1U) {
            x2 ^= context->x2_init[i];
        }
    }

    return x2;
}

static uint32_t sequence_get_x1_init(cuprn_context_t *context) {
    return context->x1_init;
}

static uint32_t sequence_get_x2_init(cuprn_context_t *context, uint32_t seed) {
    uint32_t x2 = 0;

    for (uint32_t i = 0; i < SEQUENCE_SEED_LEN; i++) {
        if ((seed >> i) & 1U) {
            x2 ^= context->x2_init[i];
        }
    }

    return x2;
}

void srslte_sequence_apply_f(cuprn_context_t *context, const float *in, float *out, uint32_t length, uint32_t seed) {
    uint32_t x1 = sequence_get_x1_init(context);       // X1 initial state is fix
    uint32_t x2 = sequence_get_x2_init(context, seed); // loads x2 initial state

    uint32_t i = 0;

    if (length >= SEQUENCE_PAR_BITS) {
        for (; i < length - (SEQUENCE_PAR_BITS - 1); i += SEQUENCE_PAR_BITS) {
            uint32_t c = (uint32_t) (x1 ^ x2);

            uint32_t j = 0;

            for (; j < SEQUENCE_PAR_BITS; j++) {
                ((uint32_t *) out)[i + j] = ((uint32_t *) in)[i] ^ (((c >> j) & 1U) << 31U);
            }

            // Step sequences
            x1 = sequence_gen_LTE_pr_memless_step_par_x1(x1);
            x2 = sequence_gen_LTE_pr_memless_step_par_x2(x2);
        }
    }

    for (; i < length; i++) {

        ((uint32_t *) out)[i] = ((uint32_t *) in)[i] ^ (((x1 ^ x2) & 1U) << 31U);

        // Step sequences
        x1 = sequence_gen_LTE_pr_memless_step_x1(x1);
        x2 = sequence_gen_LTE_pr_memless_step_x2(x2);
    }
}

void cuprn_sequence_gen_host_f(cuprn_context_t *context, float amplitude, float *out, uint32_t length, uint32_t seed) {
    uint32_t x1 = sequence_get_x1_init(context);       // X1 initial state is fix
    uint32_t x2 = sequence_get_x2_init(context, seed); // loads x2 initial state

    uint32_t i = 0;
    uint32_t a = *((uint32_t *) &amplitude);

    if (length >= SEQUENCE_PAR_BITS) {
        for (; i < length - (SEQUENCE_PAR_BITS - 1); i += SEQUENCE_PAR_BITS) {
            uint32_t c = (uint32_t) (x1 ^ x2);

            uint32_t j = 0;

            for (; j < SEQUENCE_PAR_BITS; j++) {
                ((uint32_t *) out)[i + j] = a ^ (((c >> j) & 1U) << 31U);
            }

            // Step sequences
            x1 = sequence_gen_LTE_pr_memless_step_par_x1(x1);
            x2 = sequence_gen_LTE_pr_memless_step_par_x2(x2);
        }
    }

    for (; i < length; i++) {

        ((uint32_t *) out)[i] = a ^ (((x1 ^ x2) & 1U) << 31U);

        // Step sequences
        x1 = sequence_gen_LTE_pr_memless_step_x1(x1);
        x2 = sequence_gen_LTE_pr_memless_step_x2(x2);
    }
}

__global__ void
cuprn_sequence_gen_device_f(cuprn_context_t *context, float amplitude, float *out, uint32_t length, uint32_t seed) {
    int thid = blockDim.x * blockIdx.x + threadIdx.x;

    if (thid != 0) {
        return;
    }

    uint32_t x1 = cuprn_x1_init(context);       // X1 initial state is fix
    uint32_t x2 = cuprn_x2_init(context, seed); // loads x2 initial state

    uint32_t i = 0;
    uint32_t a = *((uint32_t *) &amplitude);

    for (; i < length; i++) {

        ((uint32_t *) out)[i] = a ^ (((x1 ^ x2) & 1U) << 31U);

        // Step sequences
        x1 = cuprn_step_x1(x1);
        x2 = cuprn_step_x2(x2);
    }
}

__device__ float
cuprn_sequence_correlate_device_f(const cuprn_context_t *context, const float *input, uint32_t length,
                                  uint32_t seed) {
    float corr = 0.0f;
    float power = 0.0f;

    uint32_t x1 = cuprn_x1_init(context);       // X1 initial state is fix
    uint32_t x2 = cuprn_x2_init(context, seed); // loads x2 initial state
    uint32_t i = 0;

    if (length >= SEQUENCE_PAR_BITS) {
        for (; i < length - (SEQUENCE_PAR_BITS - 1); i += SEQUENCE_PAR_BITS) {
            uint32_t c = (uint32_t) (x1 ^ x2);

            uint32_t j = 0;

            for (; j < SEQUENCE_PAR_BITS; j++) {
                float value = input[i + j];
                power += fabsf(value);

                *((uint32_t *) &value) ^= (((c >> j) & 1U) << 31U);

                corr += value;
            }

            // Step sequences
            x1 = cuprn_step_x1_par(x1);
            x2 = cuprn_step_x2_par(x2);
        }
    }

    for (; i < length; i++) {
        float value = input[i];
        power += fabsf(value);

        *((uint32_t *) &value) ^= (((x1 ^ x2) & 1U) << 31U);

        corr += value;

        // Step sequences
        x1 = cuprn_step_x1(x1);
        x2 = cuprn_step_x2(x2);
    }

    return corr / power;
}

__device__ float
cuprn_sequence_correlate_device_s(const cuprn_context_t *context, const int16_t *input, uint32_t length,
                                  uint32_t seed) {
    int32_t corr = 0.0f;
    uint32_t power = 0.0f;

    uint32_t x1 = cuprn_x1_init(context);       // X1 initial state is fix
    uint32_t x2 = cuprn_x2_init(context, seed); // loads x2 initial state
    uint32_t i = 0;

    if (length >= SEQUENCE_PAR_BITS) {
        for (; i < length - (SEQUENCE_PAR_BITS - 1); i += SEQUENCE_PAR_BITS) {
            uint32_t c = (uint32_t) (x1 ^ x2);

            uint32_t j = 0;

            for (; j < SEQUENCE_PAR_BITS; j++) {
                int16_t value = input[i + j];
                power += abs(value);

                corr += ((c >> j) & 1U) ? -value : value;
            }

            // Step sequences
            x1 = cuprn_step_x1_par(x1);
            x2 = cuprn_step_x2_par(x2);
        }
    }

    for (; i < length; i++) {
        int16_t value = input[i];
        power += abs(value);

        corr += ((x1 ^ x2) & 1U) ? -value : value;

        // Step sequences
        x1 = cuprn_step_x1(x1);
        x2 = cuprn_step_x2(x2);
    }

    return (float) corr / (float) power;
}

cuprn_result_t *cuprn_result_init_device(uint32_t nof_blocks) {
    return cuprn_device_malloc_res(nof_blocks);
}

cuprn_result_t *cuprn_result_init_host(uint32_t nof_blocks) {
    return cuprn_host_malloc_res(nof_blocks);
}


void cuprn_result_device_free(cuprn_result_t *res) {
    cuprn_device_free_res(res);
}

void cuprn_result_host_free(cuprn_result_t *res) {
    cuprn_host_free_res(res);
}

void cuprn_result_device_2_host(cuprn_result_t *dst, const cuprn_result_t *src, uint32_t nof_blocks) {
    cuprn_copy_device_2_host_res(dst, src, nof_blocks);
}

