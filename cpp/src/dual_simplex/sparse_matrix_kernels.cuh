/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <dual_simplex/cusparse_info.hpp>
#include <dual_simplex/device_sparse_matrix.cuh>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
__global__ void scale_columns_kernel(csc_view_t<i_t, f_t> csc, raft::device_span<const f_t> scale)
{
  i_t j         = blockIdx.x;
  i_t col_start = csc.col_start[j];
  i_t col_end   = csc.col_start[j + 1];

  for (i_t p = threadIdx.x; p < col_end - col_start; p += blockDim.x) {
    csc.x[col_start + p] *= scale[j];
  }
}

template <typename i_t, typename f_t>
void initialize_cusparse_data(raft::handle_t const* handle,
                              device_csr_matrix_t<i_t, f_t>& A,
                              device_csc_matrix_t<i_t, f_t>& DAT,
                              device_csr_matrix_t<i_t, f_t>& ADAT,
                              cusparse_info_t<i_t, f_t>& cusparse_data)
{
  auto A_nnz   = A.row_start.element(A.m, A.row_start.stream());
  auto DAT_nnz = DAT.col_start.element(DAT.n, DAT.col_start.stream());

  // Create matrix descriptors
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
    &cusparse_data.matA_descr, A.m, A.n, A_nnz, A.row_start.data(), A.j.data(), A.x.data()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&cusparse_data.matDAT_descr,
                                                            DAT.n,
                                                            DAT.m,
                                                            DAT_nnz,
                                                            DAT.col_start.data(),
                                                            DAT.i.data(),
                                                            DAT.x.data()));

  // std::cout << "ADAT.m " << ADAT.m << " ADAT.n " << ADAT.n << std::endl;
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(&cusparse_data.matADAT_descr,
                                                            ADAT.m,
                                                            ADAT.n,
                                                            0,
                                                            ADAT.row_start.data(),
                                                            ADAT.j.data(),
                                                            ADAT.x.data()));
  RAFT_CUSPARSE_TRY(cusparseSpGEMM_createDescr(&cusparse_data.spgemm_descr));

  // Buffer size
  RAFT_CUSPARSE_TRY(cusparseSpGEMMreuse_workEstimation(handle->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       // cusparse_data.alpha.data(),
                                                       cusparse_data.matA_descr,
                                                       cusparse_data.matDAT_descr,
                                                       // cusparse_data.beta.data(),
                                                       cusparse_data.matADAT_descr,
                                                       //  CUDA_R_64F,
                                                       CUSPARSE_SPGEMM_DEFAULT,
                                                       cusparse_data.spgemm_descr,
                                                       &cusparse_data.buffer_size_size,
                                                       nullptr));
  cusparse_data.buffer_size.resize(cusparse_data.buffer_size_size, handle->get_stream());

  // std::cout << "buffer_size " << buffer_size << std::endl;
  RAFT_CUSPARSE_TRY(cusparseSpGEMMreuse_workEstimation(handle->get_cusparse_handle(),
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       // cusparse_data.alpha.data(),
                                                       cusparse_data.matA_descr,
                                                       cusparse_data.matDAT_descr,
                                                       // cusparse_data.beta.data(),
                                                       cusparse_data.matADAT_descr,
                                                       // CUDA_R_64F,
                                                       CUSPARSE_SPGEMM_DEFAULT,
                                                       cusparse_data.spgemm_descr,
                                                       &cusparse_data.buffer_size_size,
                                                       cusparse_data.buffer_size.data()));

  RAFT_CUSPARSE_TRY(cusparseSpGEMMreuse_nnz(handle->get_cusparse_handle(),
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            cusparse_data.matA_descr,
                                            cusparse_data.matDAT_descr,
                                            cusparse_data.matADAT_descr,
                                            CUSPARSE_SPGEMM_DEFAULT,
                                            cusparse_data.spgemm_descr,
                                            &cusparse_data.buffer_size_2_size,
                                            nullptr,
                                            &cusparse_data.buffer_size_3_size,
                                            nullptr,
                                            &cusparse_data.buffer_size_4_size,
                                            nullptr));

  cusparse_data.buffer_size_2.resize(cusparse_data.buffer_size_2_size, handle->get_stream());
  cusparse_data.buffer_size_3.resize(cusparse_data.buffer_size_3_size, handle->get_stream());
  cusparse_data.buffer_size_4.resize(cusparse_data.buffer_size_4_size, handle->get_stream());

  RAFT_CUSPARSE_TRY(cusparseSpGEMMreuse_nnz(handle->get_cusparse_handle(),
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            cusparse_data.matA_descr,
                                            cusparse_data.matDAT_descr,
                                            cusparse_data.matADAT_descr,
                                            CUSPARSE_SPGEMM_DEFAULT,
                                            cusparse_data.spgemm_descr,
                                            &cusparse_data.buffer_size_2_size,
                                            cusparse_data.buffer_size_2.data(),
                                            &cusparse_data.buffer_size_3_size,
                                            cusparse_data.buffer_size_3.data(),
                                            &cusparse_data.buffer_size_4_size,
                                            cusparse_data.buffer_size_4.data()));

  cusparse_data.buffer_size.resize(0, handle->get_stream());
  cusparse_data.buffer_size_2.resize(0, handle->get_stream());

  // get matrix C non-zero entries C_nnz1
  int64_t ADAT_num_rows, ADAT_num_cols, ADAT_nnz1;
  RAFT_CUSPARSE_TRY(
    cusparseSpMatGetSize(cusparse_data.matADAT_descr, &ADAT_num_rows, &ADAT_num_cols, &ADAT_nnz1));
  // std::cout << "ADAT_num_rows " << ADAT_num_rows << " ADAT_num_cols " << ADAT_num_cols <<
  // std::endl; std::cout << "ADAT_nnz1 " << ADAT_nnz1 << std::endl;
  ADAT.resize_to_nnz(ADAT_nnz1, handle->get_stream());

  // update matC with the new pointers
  RAFT_CUSPARSE_TRY(cusparseCsrSetPointers(
    cusparse_data.matADAT_descr, ADAT.row_start.data(), ADAT.j.data(), ADAT.x.data()));

  RAFT_CUSPARSE_TRY(cusparseSpGEMMreuse_copy(handle->get_cusparse_handle(),
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             cusparse_data.matA_descr,
                                             cusparse_data.matDAT_descr,
                                             cusparse_data.matADAT_descr,
                                             CUSPARSE_SPGEMM_DEFAULT,
                                             cusparse_data.spgemm_descr,
                                             &cusparse_data.buffer_size_5_size,
                                             nullptr));
  cusparse_data.buffer_size_5.resize(cusparse_data.buffer_size_5_size, handle->get_stream());
  RAFT_CUSPARSE_TRY(cusparseSpGEMMreuse_copy(handle->get_cusparse_handle(),
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             cusparse_data.matA_descr,
                                             cusparse_data.matDAT_descr,
                                             cusparse_data.matADAT_descr,
                                             CUSPARSE_SPGEMM_DEFAULT,
                                             cusparse_data.spgemm_descr,
                                             &cusparse_data.buffer_size_5_size,
                                             cusparse_data.buffer_size_5.data()));
  cusparse_data.buffer_size_3.resize(0, handle->get_stream());
}

template <typename i_t, typename f_t>
void multiply_kernels(raft::handle_t const* handle,
                      device_csr_matrix_t<i_t, f_t>& A,
                      device_csc_matrix_t<i_t, f_t>& DAT,
                      device_csr_matrix_t<i_t, f_t>& ADAT,
                      cusparse_info_t<i_t, f_t>& cusparse_data)
{
  thrust::fill(
    rmm::exec_policy(handle->get_stream()), ADAT.x.data(), ADAT.x.data() + ADAT.x.size(), 0.0);

  RAFT_CUSPARSE_TRY(
    cusparseSpGEMMreuse_compute(handle->get_cusparse_handle(),
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                cusparse_data.alpha.data(),
                                cusparse_data.matA_descr,    // non-const descriptor supported
                                cusparse_data.matDAT_descr,  // non-const descriptor supported
                                cusparse_data.beta.data(),
                                cusparse_data.matADAT_descr,
                                CUDA_R_64F,
                                CUSPARSE_SPGEMM_DEFAULT,
                                cusparse_data.spgemm_descr));
}

}  // namespace cuopt::linear_programming::dual_simplex
