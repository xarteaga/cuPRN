#
# \section COPYRIGHT
#
# Copyright 2020-2021 Xavier Arteaga
#
# By using this file, you agree to the terms and conditions set
# forth in the LICENSE file which can be found at the top level of
# the distribution.
#

cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_C_COMPILER /usr/bin/gcc-7)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_PROPAGATE_HOST_FLAGS ON)

project(cuPRN CUDA)

add_executable(cuPRN cuprn.cu vector.cu sequence.cu)

set_target_properties(
        cuPRN
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)