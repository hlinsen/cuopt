# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(DUAL_SIMPLEX_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/barrier.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/basis_solves.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/basis_updates.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/bound_flipping_ratio_test.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/branch_and_bound.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/crossover.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/initial_basis.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mip_node.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/phase1.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/phase2.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/presolve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/primal.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/pseudo_costs.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/right_looking_lu.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/scaling.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/singletons.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_matrix.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_vector.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tic_toc.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/triangle_solve.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/vector_math.cu)

set_source_files_properties(${DUAL_SIMPLEX_SRC_FILES} DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTIES COMPILE_OPTIONS "-g1")

set(CUOPT_SRC_FILES ${CUOPT_SRC_FILES}
  ${DUAL_SIMPLEX_SRC_FILES} PARENT_SCOPE)
