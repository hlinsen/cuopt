#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Clang-host C++ build dependencies"

clang_env_file="${RUNNER_TEMP:-/tmp}/cuopt-clang-build-env.yaml"
rapids-dependency-file-generator \
  --output conda \
  --file-key build_cpp_clang_host \
  --matrix "cuda=13.3;arch=$(arch)" \
  | tee "${clang_env_file}"

rapids-logger "Create Clang-host C++ build environment"
rapids-mamba-retry env create --yes -f "${clang_env_file}" -n clang-build

set +u
conda activate clang-build
set -u

source rapids-configure-sccache

export CMAKE_GENERATOR=Ninja
CC="$(command -v clang)"
CXX="$(command -v clang++)"
export CC CXX
export CUDAHOSTCXX="${CXX}"
export LIBCUOPT_BUILD_DIR="${PWD}/cpp/build-clang"

rapids-print-env
clang --version
nvcc --version

rapids-logger "Build C++ and CUDA targets with Clang as the NVCC host compiler"
sccache --zero-stats

# One representative GPU architecture keeps this portability gate smaller than the package build.
./build.sh -v libcuopt --ci-only-arch --cache-tool=sccache \
  --cmake-args=\"-DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX}\"

grep -Eq '^CMAKE_CXX_COMPILER:FILEPATH=.*/clang\+\+$' "${LIBCUOPT_BUILD_DIR}/CMakeCache.txt"
grep -Eq '^CMAKE_CUDA_COMPILER:FILEPATH=.*/nvcc$' "${LIBCUOPT_BUILD_DIR}/CMakeCache.txt"
grep -Eq '^CMAKE_CUDA_HOST_COMPILER:FILEPATH=.*/clang\+\+$' \
  "${LIBCUOPT_BUILD_DIR}/CMakeCache.txt"

sccache --show-adv-stats
