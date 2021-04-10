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

#ifndef CUPRN_MEMORY_CUH
#define CUPRN_MEMORY_CUH

#define CUPRN_MEMORY_PROTO(TYPE, NAME)                                                                                 \
  TYPE* cuprn_device_malloc_##NAME(size_t size);                                                                       \
  void  cuprn_device_free_##NAME(TYPE* p);                                                                             \
  TYPE* cuprn_host_malloc_##NAME(size_t size);                                                                         \
  void  cuprn_host_free_##NAME(TYPE* p);                                                                               \
  void  cuprn_copy_device_2_host_##NAME(TYPE* dst, const TYPE* src, size_t size);                                      \
  void  cuprn_copy_host_2_device_##NAME(TYPE* dst, const TYPE* src, size_t size);                                      \
  void  cuprn_copy_device_2_device_##NAME(TYPE* dst, const TYPE* src, size_t size);

#define CUPRN_MEMORY(TYPE, NAME)                                                                                       \
  TYPE* cuprn_device_malloc_##NAME(size_t size)                                                                        \
  {                                                                                                                    \
    TYPE* ptr = NULL;                                                                                                  \
    if (cudaMalloc((void**)&ptr, sizeof(TYPE) * size) != cudaSuccess) {                                                \
      return NULL;                                                                                                     \
    }                                                                                                                  \
    return ptr;                                                                                                        \
  }                                                                                                                    \
                                                                                                                       \
  void cuprn_device_free_##NAME(TYPE* p)                                                                               \
  {                                                                                                                    \
    if (p) {                                                                                                           \
      cudaFree((void*)p);                                                                                              \
    }                                                                                                                  \
  }                                                                                                                    \
                                                                                                                       \
  TYPE* cuprn_host_malloc_##NAME(size_t size) { return (TYPE*)malloc(sizeof(TYPE) * size); }                           \
  void  cuprn_host_free_##NAME(TYPE* p)                                                                                \
  {                                                                                                                    \
    if (p) {                                                                                                           \
      free((void*)p);                                                                                                  \
    }                                                                                                                  \
  }                                                                                                                    \
  void cuprn_copy_device_2_host_##NAME(TYPE* dst, const TYPE* src, size_t size)                                        \
  {                                                                                                                    \
    cudaMemcpy(dst, src, size * sizeof(TYPE), cudaMemcpyDeviceToHost);                                                 \
  }                                                                                                                    \
  void cuprn_copy_host_2_device_##NAME(TYPE* dst, const TYPE* src, size_t size)                                        \
  {                                                                                                                    \
    cudaMemcpy(dst, src, size * sizeof(TYPE), cudaMemcpyHostToDevice);                                                 \
  }                                                                                                                    \
  void cuprn_copy_device_2_device_##NAME(TYPE* dst, const TYPE* src, size_t size)                                      \
  {                                                                                                                    \
    cudaMemcpy(dst, src, size * sizeof(TYPE), cudaMemcpyDeviceToDevice);                                               \
  }

#endif // CUPRN_MEMORY_CUH
