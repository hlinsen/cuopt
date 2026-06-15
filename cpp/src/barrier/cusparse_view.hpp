/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/sparse_matrix.hpp>

#include <pdlp/cusparse_view.hpp>

#include <cusparse_v2.h>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/core/handle.hpp>

// Lightweight cuSparse view
// Only owns data linked to the associated matrix
// Associated dense vector should be owned by the calling object
// This allows handling many different X Y vector along with one common matrix
namespace cuopt::linear_programming::dual_simplex {
template <typename i_t, typename f_t>
class cusparse_view_t {
 public:
  // TMP matrix data should already be on the GPU and in CSR not CSC
  cusparse_view_t(raft::handle_t const* handle_ptr, const csc_matrix_t<i_t, f_t>& A);
  ~cusparse_view_t();

  detail::cusparse_dn_vec_descr_wrapper_t<f_t> create_vector(rmm::device_uvector<f_t> const& vec);

  template <typename AllocatorA, typename AllocatorB>
  void spmv(f_t alpha,
            const std::vector<f_t, AllocatorA>& x,
            f_t beta,
            std::vector<f_t, AllocatorB>& y);
  void spmv(f_t alpha, rmm::device_uvector<f_t> const& x, f_t beta, rmm::device_uvector<f_t>& y);
  void spmv(f_t alpha,
            detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
            f_t beta,
            detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& y);
  template <typename AllocatorA, typename AllocatorB>
  void transpose_spmv(f_t alpha,
                      const std::vector<f_t, AllocatorA>& x,
                      f_t beta,
                      std::vector<f_t, AllocatorB>& y);
  void transpose_spmv(f_t alpha,
                      rmm::device_uvector<f_t> const& x,
                      f_t beta,
                      rmm::device_uvector<f_t>& y);
  void transpose_spmv(f_t alpha,
                      detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& x,
                      f_t beta,
                      detail::cusparse_dn_vec_descr_wrapper_t<f_t> const& y);

  raft::handle_t const* handle_ptr_{nullptr};

 private:
  rmm::device_uvector<i_t> A_offsets_;
  rmm::device_uvector<i_t> A_indices_;
  rmm::device_uvector<f_t> A_data_;
  cusparseSpMatDescr_t A_;
  rmm::device_uvector<i_t> A_T_offsets_;
  rmm::device_uvector<i_t> A_T_indices_;
  rmm::device_uvector<f_t> A_T_data_;
  cusparseSpMatDescr_t A_T_;
  rmm::device_buffer spmv_buffer_;
  rmm::device_buffer spmv_buffer_transpose_;
  rmm::device_scalar<f_t> d_one_;
  rmm::device_scalar<f_t> d_minus_one_;
  rmm::device_scalar<f_t> d_zero_;

  // Fused SpMVOp acceleration. A plan is built once per matrix (A and A_T) and reused across all
  // barrier iterations, which is cheaper than the generic cusparseSpMV path. This mirrors the
  // SpMVOp speed-up adopted by PDLP. When the runtime cuSPARSE lacks the SpMVOp symbols (or f_t is
  // not double) spmvop_enabled_ stays false and spmv()/transpose_spmv() fall back to cusparseSpMV.
  bool spmvop_enabled_{false};
#if CUOPT_CUSPARSE_VER_12_7_UP
  rmm::device_uvector<uint8_t> spmvop_buffer_;
  rmm::device_uvector<uint8_t> spmvop_buffer_transpose_;
  // descr declared before plan so the destructor tears the plan down first
  detail::cusparse_spmvop_descr_wrapper_t spmv_op_descr_A_;
  detail::cusparse_spmvop_plan_wrapper_t spmv_op_plan_A_;
  detail::cusparse_spmvop_descr_wrapper_t spmv_op_descr_A_T_;
  detail::cusparse_spmvop_plan_wrapper_t spmv_op_plan_A_T_;
#endif  // CUOPT_CUSPARSE_VER_12_7_UP
};
}  // namespace cuopt::linear_programming::dual_simplex
