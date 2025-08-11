#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

source rapids-init-pip

package_name="libcuopt"
package_dir="python/libcuopt"

# Install Boost and TBB
bash ci/utils/install_boost_tbb.sh

# BOOST_VERSION="1.85"
# BOOST_INSTALL_DIR="/opt/boost-${BOOST_VERSION}"
# BOOST_SRC_DIR="/tmp/boost"

# # Remove any existing Boost installation at the target location
# if [ -d "${BOOST_INSTALL_DIR}" ]; then
#     echo "Removing existing Boost at ${BOOST_INSTALL_DIR}"
#     rm -rf "${BOOST_INSTALL_DIR}"
# fi

# # Clean up any previous source directory
# if [ -d "${BOOST_SRC_DIR}" ]; then
#     echo "Removing previous Boost source at ${BOOST_SRC_DIR}"
#     rm -rf "${BOOST_SRC_DIR}"
# fi

# echo "Cloning Boost ${BOOST_VERSION}..."
# git clone --branch "boost-${BOOST_VERSION}.0" --depth 1 --recurse-submodules https://github.com/boostorg/boost.git "${BOOST_SRC_DIR}"

# pushd "${BOOST_SRC_DIR}"

# echo "Bootstrapping Boost..."
# ./bootstrap.sh

# echo "Installing Boost to ${BOOST_INSTALL_DIR}..."
# ./b2 install --prefix="${BOOST_INSTALL_DIR}" -j"$(nproc)" \
#     --with-filesystem \
#     --with-regex \
#     --with-log \
#     --with-thread \
#     --with-system \
#     --with-iostreams \
#     --with-serialization \
#     --with-program_options

# popd

# echo "Cleaning up Boost source directory..."
# rm -rf "${BOOST_SRC_DIR}"

export SKBUILD_CMAKE_ARGS="-DCUOPT_BUILD_WHEELS=ON;-DDISABLE_DEPRECATION_WARNING=ON"

# For pull requests we are enabling assert mode.
if [ "$RAPIDS_BUILD_TYPE" = "pull-request" ]; then
    echo "Building in assert mode"
    export SKBUILD_CMAKE_ARGS="${SKBUILD_CMAKE_ARGS};-DDEFINE_ASSERT=True"
else
    echo "Building in release mode"
fi

rapids-logger "Generating build requirements"

CUOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
echo "cuopt-mps-parser @ file://$(echo ${CUOPT_MPS_PARSER_WHEELHOUSE}/cuopt_mps_parser*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0


EXCLUDE_ARGS=(
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink*.so*"
  --exclude "librapids_logger.so"
  --exclude "libmps_parser.so"
  --exclude "librmm.so"
)

ci/build_wheel.sh libcuopt ${package_dir}

mkdir -p final_dist
python -m auditwheel repair "${EXCLUDE_ARGS[@]}" -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" ${package_dir}/dist/*

ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
