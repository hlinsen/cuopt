/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/barrier.hpp>

#include <dual_simplex/cusparse_info.hpp>
#include <dual_simplex/dense_matrix.hpp>
#include <dual_simplex/dense_vector.hpp>
#include <dual_simplex/device_sparse_matrix.cuh>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_cholesky.cuh>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/sparse_matrix_kernels.cuh>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/vector_math.cuh>
#include "dual_simplex/cusparse_view.hpp"

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>

#include <numeric>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/linalg/dot.cuh>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

namespace cuopt::linear_programming::dual_simplex {

auto constexpr use_gpu = true;

template <typename i_t, typename f_t>
class iteration_data_t {
 public:
  iteration_data_t(const lp_problem_t<i_t, f_t>& lp,
                   i_t num_upper_bounds,
                   const simplex_solver_settings_t<i_t, f_t>& settings)
    : upper_bounds(num_upper_bounds),
      c(lp.objective),
      b(lp.rhs),
      w(num_upper_bounds),
      x(lp.num_cols),
      y(lp.num_rows),
      v(num_upper_bounds),
      z(lp.num_cols),
      w_save(num_upper_bounds),
      x_save(lp.num_cols),
      y_save(lp.num_rows),
      v_save(num_upper_bounds),
      z_save(lp.num_cols),
      diag(lp.num_cols),
      inv_diag(lp.num_cols),
      inv_sqrt_diag(lp.num_cols),
      AD(lp.num_cols, lp.num_rows, 0),
      AT(lp.num_rows, lp.num_cols, 0),
      ADAT(lp.num_rows, lp.num_rows, 0),
      A_dense(lp.num_rows, 0),
      AD_dense(0, 0),
      H(0, 0),
      Hchol(0, 0),
      A(lp.A),
      primal_residual(lp.num_rows),
      bound_residual(num_upper_bounds),
      dual_residual(lp.num_cols),
      complementarity_xz_residual(lp.num_cols),
      complementarity_wv_residual(num_upper_bounds),
      cusparse_view_(lp.handle_ptr,
                     lp.A.m,
                     lp.A.n,
                     lp.A.x.size(),
                     lp.A.col_start,
                     lp.A.i,
                     lp.A.x  // TMP we should pass a GPU CSR view directly
                     ),
      primal_rhs(lp.num_rows),
      bound_rhs(num_upper_bounds),
      dual_rhs(lp.num_cols),
      complementarity_xz_rhs(lp.num_cols),
      complementarity_wv_rhs(num_upper_bounds),
      dw_aff(num_upper_bounds),
      dx_aff(lp.num_cols),
      dy_aff(lp.num_rows),
      dv_aff(num_upper_bounds),
      dz_aff(lp.num_cols),
      dw(num_upper_bounds),
      dx(lp.num_cols),
      dy(lp.num_rows),
      dv(num_upper_bounds),
      dz(lp.num_cols),
      cusparse_info(lp.handle_ptr),
      device_AD(lp.num_cols, lp.num_rows, 0, lp.handle_ptr->get_stream()),
      device_A(lp.num_cols, lp.num_rows, 0, lp.handle_ptr->get_stream()),
      device_ADAT(lp.num_rows, lp.num_rows, 0, lp.handle_ptr->get_stream()),
      d_original_A_values(0, lp.handle_ptr->get_stream()),
      device_A_x_values(0, lp.handle_ptr->get_stream()),
      d_inv_diag_prime(0, lp.handle_ptr->get_stream()),
      d_flag_buffer(0, lp.handle_ptr->get_stream()),
      d_num_flag(lp.handle_ptr->get_stream()),
      d_inv_diag(lp.num_cols, lp.handle_ptr->get_stream()),
      d_cols_to_remove(0, lp.handle_ptr->get_stream()),
      has_factorization(false),
      num_factorizations(0),
      has_solve_info(false),
      settings_(settings),
      handle_ptr(lp.handle_ptr),
      stream_view_(lp.handle_ptr->get_stream()),
      transform_reduce_helper_(lp.handle_ptr->get_stream()),
      sum_reduce_helper_(lp.handle_ptr->get_stream())
  {
    raft::common::nvtx::range fun_scope("Barrier: LP Data Creation");

    // Allocating GPU flag data for Form ADAT
    if (use_gpu) {
      raft::common::nvtx::range fun_scope("Barrier: GPU Flag memory allocation");

      cub::DeviceSelect::Flagged(
        nullptr,
        flag_buffer_size,
        d_inv_diag_prime.data(),  // Not the actual input but just to allcoate the memory
        thrust::make_transform_iterator(d_cols_to_remove.data(), cuda::std::logical_not<i_t>{}),
        d_inv_diag_prime.data(),
        d_num_flag.data(),
        inv_diag.size());

      d_flag_buffer.resize(flag_buffer_size, stream_view_);
    }
    // Create the upper bounds vector
    n_upper_bounds = 0;
    for (i_t j = 0; j < lp.num_cols; j++) {
      if (lp.upper[j] < inf) { upper_bounds[n_upper_bounds++] = j; }
    }
    settings.log.printf("n_upper_bounds %d\n", n_upper_bounds);

    diag.set_scalar(1.0);
    if (n_upper_bounds > 0) {
      for (i_t k = 0; k < n_upper_bounds; k++) {
        i_t j   = upper_bounds[k];
        diag[j] = 2.0;
      }
    }
    inv_diag.set_scalar(1.0);
    if (n_upper_bounds > 0) { diag.inverse(inv_diag); }
    if (use_gpu) {
      // TMP diag and inv_diag should directly created and filled on the GPU
      raft::copy(d_inv_diag.data(), inv_diag.data(), inv_diag.size(), stream_view_);
    }
    inv_sqrt_diag.set_scalar(1.0);
    if (n_upper_bounds > 0) { inv_diag.sqrt(inv_sqrt_diag); }

    n_dense_columns = 0;
    std::vector<i_t> dense_columns_unordered;
    if (settings.eliminate_dense_columns) {
      f_t start_column_density = tic();
      find_dense_columns(lp.A, settings, dense_columns_unordered);
      for (i_t j : dense_columns_unordered) {
        settings.log.printf("Dense column %6d\n", j);
      }
      float64_t column_density_time = toc(start_column_density);
      n_dense_columns               = static_cast<i_t>(dense_columns_unordered.size());
      settings.log.printf(
        "Found %d dense columns in %.2fs\n", n_dense_columns, column_density_time);
    }

    // Copy A into AD
    AD = lp.A;
    if (n_dense_columns > 0) {
      cols_to_remove.resize(lp.num_cols, 0);
      for (i_t k : dense_columns_unordered) {
        cols_to_remove[k] = 1;
      }
      d_cols_to_remove.resize(cols_to_remove.size(), stream_view_);
      raft::copy(
        d_cols_to_remove.data(), cols_to_remove.data(), cols_to_remove.size(), stream_view_);
      dense_columns.clear();
      dense_columns.reserve(n_dense_columns);
      for (i_t j = 0; j < lp.num_cols; j++) {
        if (cols_to_remove[j]) { dense_columns.push_back(j); }
      }
      AD.remove_columns(cols_to_remove);

      sparse_mark.resize(lp.num_cols, 1);
      for (i_t k : dense_columns) {
        sparse_mark[k] = 0;
      }

      A_dense.resize(AD.m, n_dense_columns);
      i_t k = 0;
      for (i_t j : dense_columns) {
        A_dense.from_sparse(lp.A, j, k++);
      }
    }
    original_A_values = AD.x;
    if (use_gpu) {
      d_original_A_values.resize(original_A_values.size(), handle_ptr->get_stream());
      raft::copy(d_original_A_values.data(), AD.x.data(), AD.x.size(), handle_ptr->get_stream());
    }
    AD.transpose(AT);

    if (use_gpu) {
      device_AD.copy(AD, handle_ptr->get_stream());
      // For efficient scaling of AD col we form the col index array
      device_AD.form_col_index(handle_ptr->get_stream());
      device_A_x_values.resize(original_A_values.size(), handle_ptr->get_stream());
      raft::copy(
        device_A_x_values.data(), device_AD.x.data(), device_AD.x.size(), handle_ptr->get_stream());
      csr_matrix_t<i_t, f_t> host_A_CSR(1, 1, 1);  // Sizes will be set by to_compressed_row()
      AD.to_compressed_row(host_A_CSR);
      device_A.copy(host_A_CSR, lp.handle_ptr->get_stream());
      RAFT_CHECK_CUDA(handle_ptr->get_stream());
    }
    form_adat(true);

    chol = std::make_unique<sparse_cholesky_cudss_t<i_t, f_t>>(
      settings, lp.num_rows, handle_ptr->get_stream());
    chol->set_positive_definite(false);

    // Perform symbolic analysis
    if (use_gpu) {
      chol->analyze(device_ADAT);
    } else {
      chol->analyze(ADAT);
    }
  }

  void form_adat(bool first_call = false)
  {
    handle_ptr->sync_stream();
    raft::common::nvtx::range fun_scope("Barrier: Form ADAT");
    float64_t start_form_adat = tic();
    const i_t m               = AD.m;

    if (use_gpu) {
      // TODO do we really need this copy? (it's ok since gpu to gpu)
      raft::copy(device_AD.x.data(),
                 d_original_A_values.data(),
                 d_original_A_values.size(),
                 handle_ptr->get_stream());
      if (n_dense_columns > 0) {
        // Adjust inv_diag
        d_inv_diag_prime.resize(AD.n, stream_view_);
        // Copy If
        cub::DeviceSelect::Flagged(
          d_flag_buffer.data(),
          flag_buffer_size,
          d_inv_diag.data(),
          thrust::make_transform_iterator(d_cols_to_remove.data(), cuda::std::logical_not<i_t>{}),
          d_inv_diag_prime.data(),
          d_num_flag.data(),
          d_inv_diag.size());
      } else {
        d_inv_diag_prime.resize(inv_diag.size(), stream_view_);
        raft::copy(d_inv_diag_prime.data(), d_inv_diag.data(), inv_diag.size(), stream_view_);
      }

      if (d_inv_diag_prime.size() != AD.n) {
        settings_.log.printf(
          "inv_diag_prime.size() = %ld, AD.n = %d\n", d_inv_diag_prime.size(), AD.n);
        exit(1);
      }

      thrust::for_each_n(rmm::exec_policy(stream_view_),
                         thrust::make_counting_iterator<i_t>(0),
                         i_t(device_AD.x.size()),
                         [span_x       = cuopt::make_span(device_AD.x),
                          span_scale   = cuopt::make_span(d_inv_diag_prime),
                          span_col_ind = cuopt::make_span(device_AD.col_index)] __device__(i_t i) {
                           span_x[i] *= span_scale[span_col_ind[i]];
                         });

      if (first_call) {
        initialize_cusparse_data<i_t, f_t>(
          handle_ptr, device_A, device_AD, device_ADAT, cusparse_info);
      }

      // float64_t start_multiply = tic();
      // handle_ptr->sync_stream();
      multiply_kernels<i_t, f_t>(handle_ptr, device_A, device_AD, device_ADAT, cusparse_info);
      handle_ptr->sync_stream();
      // float64_t multiply_time = toc(start_multiply);
      // settings_.log.printf("multiply_time %.2fs\n", multiply_time);
      // if (0) {
      //   auto host_ADAT = device_ADAT.to_host(handle_ptr->get_stream());
      //   host_ADAT.to_compressed_col(ADAT);
      //   auto tmp_ADAT = ADAT;
      //   // AD = device_AD.to_host(handle_ptr->get_stream());
      //   AD.scale_columns(inv_diag_prime);
      //   multiply(AD, AT, ADAT);
      //   ADAT.compare(tmp_ADAT);
      // }

      auto adat_nnz       = device_ADAT.row_start.element(device_ADAT.m, handle_ptr->get_stream());
      float64_t adat_time = toc(start_form_adat);

      if (num_factorizations == 0) {
        settings_.log.printf("ADAT time %.2fs\n", adat_time);
        settings_.log.printf(
          "ADAT nonzeros %e density %.2f\n",
          static_cast<float64_t>(adat_nnz),
          static_cast<float64_t>(adat_nnz) /
            (static_cast<float64_t>(device_ADAT.m) * static_cast<float64_t>(device_ADAT.m)));
      }
    } else {
      // Restore the columns of AD to A
      AD.x = original_A_values;
      std::vector<f_t> inv_diag_prime;
      if (n_dense_columns > 0) {
        // Adjust inv_diag
        inv_diag_prime.resize(AD.n);
        const i_t n = A.n;

        i_t new_j = 0;
        for (i_t j = 0; j < n; j++) {
          if (cols_to_remove[j]) { continue; }
          inv_diag_prime[new_j++] = inv_diag[j];
        }
      } else {
        inv_diag_prime = copy(inv_diag);
      }

      if (inv_diag_prime.size() != AD.n) {
        settings_.log.printf(
          "inv_diag_prime.size() = %ld, AD.n = %d\n", inv_diag_prime.size(), AD.n);
        exit(1);
      }
      AD.scale_columns(inv_diag_prime);
      multiply(AD, AT, ADAT);

      float64_t adat_time = toc(start_form_adat);
      if (num_factorizations == 0) {
        settings_.log.printf("ADAT time %.2fs\n", adat_time);
        settings_.log.printf("ADAT nonzeros %e density %.2f\n",
                             static_cast<float64_t>(ADAT.col_start[m]),
                             static_cast<float64_t>(ADAT.col_start[m]) /
                               (static_cast<float64_t>(m) * static_cast<float64_t>(m)));
      }
    }
  }

  i_t solve_adat(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
  {
    if (n_dense_columns == 0) {
      // Solve ADAT * x = b
      return chol->solve(b, x);
    } else {
      // Use Sherman Morrison followed by PCG

      // ADA^T = A_sparse * D_sparse * A_sparse^T + A_dense * D_dense * A_dense^T
      // Let p be the number of dense columns
      // U = A_dense * D_dense^0.5 is m x p
      // U^T = D_dense^0.5 * A_dense^T is p x m

      // We have that A D A^T *x = b is
      // (A_sparse * D_sparse * A_sparse^T + A_dense * D_dense * A_dense^T) * x = b
      // (A_sparse * D_sparse * A_sparse^T + U * U^T ) * x = b
      // We can write this as the 2x2 system
      //
      // [ A_sparse * D_sparse * A_sparse^T     U ][ x ] = [ b ]
      // [ U^T                                  -I][ y ]   [ 0 ]
      //
      // We can write x = (A_sparse * D_sparse * A_sparse^T)^{-1} * (b - U * y)
      // So U^T * x - y = 0 or
      // U^T * (A_sparse * D_sparse * A_sparse^T)^{-1} * (b - U * y) - y = 0
      // (U^T * (A_sparse * D_sparse * A_sparse^T)^{-1} U + I) * y = U^T * (A_sparse * D_sparse *
      // A_sparse^T)^{-1} * b
      //  H * y = g
      // where H = U^T * (A_sparse * D_sparse * A_sparse^T)^{-1} U + I
      // and g = U^T * (A_sparse * D_sparse * A_sparse^T)^{-1} * b
      // Let (A_sparse * D_sparse * A_sparse^T)* w = b
      // then g = U^T * w
      // Let (A_sparse * D_sparse * A_sparse^T) * M = U
      // then H = U^T * M + I
      //
      // We can use a dense cholesky factorization of H to solve for y

      const bool debug = false;

      dense_vector_t<i_t, f_t> w(AD.m);
      i_t solve_status = chol->solve(b, w);
      if (solve_status != 0) { return solve_status; }

      if (!has_solve_info) {
        AD_dense = A_dense;

        // AD_dense = A_dense * D_dense
        dense_vector_t<i_t, f_t> dense_diag(n_dense_columns);
        i_t k = 0;
        for (i_t j : dense_columns) {
          dense_diag[k++] = std::sqrt(inv_diag[j]);
        }
        AD_dense.scale_columns(dense_diag);

        dense_matrix_t<i_t, f_t> M(AD.m, n_dense_columns);
        H.resize(n_dense_columns, n_dense_columns);
        for (i_t k = 0; k < n_dense_columns; k++) {
          dense_vector_t<i_t, f_t> U_col(AD.m);
          // U_col = AD_dense(:, k)
          for (i_t i = 0; i < AD.m; i++) {
            U_col[i] = AD_dense(i, k);
          }
          dense_vector_t<i_t, f_t> M_col(AD.m);
          solve_status = chol->solve(U_col, M_col);
          if (solve_status != 0) { return solve_status; }
          M.set_column(k, M_col);

          if (debug) {
            dense_vector_t<i_t, f_t> M_residual = U_col;
            matrix_vector_multiply(ADAT, 1.0, M_col, -1.0, M_residual);
            settings_.log.printf(
              "|| A_sparse * D_sparse * A_sparse^T * M(:, k) - AD_dense(:, k) ||_2 = %e\n",
              vector_norm2<i_t, f_t>(M_residual));
          }
        }
        // A_sparse * D_sparse * A_sparse^T * M = U = AD_dense
        // H = AD_dense^T * M
        AD_dense.transpose_matrix_multiply(1.0, M, 0.0, H);
        dense_vector_t<i_t, f_t> e(n_dense_columns);
        e.set_scalar(1.0);
        // H = AD_dense^T * M + I
        H.add_diagonal(e);

        // H = L*L^T
        Hchol.resize(n_dense_columns, n_dense_columns);  // Hcol = L
        H.chol(Hchol);
        has_solve_info = true;
      }

      dense_vector_t<i_t, f_t> g(n_dense_columns);
      // g = D_dense * A_dense^T * w
      AD_dense.transpose_multiply(1.0, w, 0.0, g);

      if (debug) {
        for (i_t k = 0; k < n_dense_columns; k++) {
          for (i_t h = 0; h < n_dense_columns; h++) {
            if (std::abs(H(k, h) - H(h, k)) > 1e-10) {
              settings_.log.printf(
                "H(%d, %d) = %e, H(%d, %d) = %e\n", k, h, H(k, h), h, k, H(h, k));
            }
          }
        }
      }

      dense_vector_t<i_t, f_t> y(n_dense_columns);
      // H *y = g
      // L*L^T * y = g
      // L*u = g
      dense_vector_t<i_t, f_t> u(n_dense_columns);
      Hchol.triangular_solve(g, u);
      // L^T y = u
      Hchol.triangular_solve_transpose(u, y);

      if (debug) {
        dense_vector_t<i_t, f_t> H_residual = g;
        H.matrix_vector_multiply(1.0, y, -1.0, H_residual);
        settings_.log.printf("|| H * y - g ||_2 = %e\n", vector_norm2<i_t, f_t>(H_residual));
      }

      // x = (A_sparse * D_sparse * A_sparse^T)^{-1} * (b - U * y)
      // v = U *y = AD_dense * y
      dense_vector_t<i_t, f_t> v(AD.m);
      AD_dense.matrix_vector_multiply(1.0, y, 0.0, v);

      // v = b - U*y
      v.axpy(1.0, b, -1.0);

      // A_sparse * D_sparse * A_sparse^T * x = v
      solve_status = chol->solve(v, x);
      if (solve_status != 0) { return solve_status; }

      if (debug) {
        dense_vector_t<i_t, f_t> solve_residual = v;
        matrix_vector_multiply(ADAT, 1.0, x, -1.0, solve_residual);
        settings_.log.printf("|| A_sparse * D * A_sparse^T * x - v ||_2 = %e\n",
                             vector_norm2<i_t, f_t>(solve_residual));
      }

      if (debug) {
        // Check U^T * x - y = 0;
        dense_vector_t<i_t, f_t> residual_2 = y;
        AD_dense.transpose_multiply(1.0, x, -1.0, residual_2);
        settings_.log.printf("|| U^T * x - y ||_2 = %e\n", vector_norm2<i_t, f_t>(residual_2));
      }

      if (debug) {
        // Check A_sparse * D_sparse * A_sparse^T * x  + U * y = b
        dense_vector_t<i_t, f_t> residual_1 = b;
        AD_dense.matrix_vector_multiply(1.0, y, -1.0, residual_1);
        matrix_vector_multiply(ADAT, 1.0, x, 1.0, residual_1);
        settings_.log.printf("|| A_sparse * D_sparse * A_sparse^T * x + U * y - b ||_2 = %e\n",
                             vector_norm2<i_t, f_t>(residual_1));
      }

      if (0 && debug) {
        csc_matrix_t<i_t, f_t> A_full_D = A;
        A_full_D.scale_columns(inv_diag);

        csc_matrix_t<i_t, f_t> A_full_D_T(A_full_D.n, A_full_D.m, 1);
        A_full_D.transpose(A_full_D_T);

        csc_matrix_t<i_t, f_t> ADAT_full(AD.m, AD.m, 1);
        multiply(A, A_full_D_T, ADAT_full);

        f_t max_error = 0.0;
        for (i_t i = 0; i < AD.m; i++) {
          dense_vector_t<i_t, f_t> ei(AD.m);
          ei.set_scalar(0.0);
          ei[i] = 1.0;

          dense_vector_t<i_t, f_t> u(AD.m);

          matrix_vector_multiply(ADAT_full, 1.0, ei, 0.0, u);

          adat_multiply(-1.0, ei, 1.0, u);

          max_error = std::max(max_error, vector_norm2<i_t, f_t>(u));
        }
        settings_.log.printf("|| ADAT(e_i) - ADA^T * e_i ||_2 = %e\n", max_error);
      }

      if (debug) {
        dense_matrix_t<i_t, f_t> UUT(AD.m, AD.m);

        for (i_t i = 0; i < AD.m; i++) {
          dense_vector_t<i_t, f_t> ei(AD.m);
          ei.set_scalar(0.0);
          ei[i] = 1.0;

          dense_vector_t<i_t, f_t> UTei(n_dense_columns);
          AD_dense.transpose_multiply(1.0, ei, 0.0, UTei);

          dense_vector_t<i_t, f_t> U_col(AD.m);
          AD_dense.matrix_vector_multiply(1.0, UTei, 0.0, U_col);

          UUT.set_column(i, U_col);
        }

        csc_matrix_t<i_t, f_t> A_dense_csc = A;
        A_dense_csc.remove_columns(sparse_mark);

        std::vector<f_t> inv_diag_prime(n_dense_columns);
        i_t k = 0;
        for (i_t j : dense_columns) {
          inv_diag_prime[k++] = std::sqrt(inv_diag[j]);
        }
        A_dense_csc.scale_columns(inv_diag_prime);

        csc_matrix_t<i_t, f_t> AT_dense_transpose(1, 1, 1);
        A_dense_csc.transpose(AT_dense_transpose);

        csc_matrix_t<i_t, f_t> ADAT_dense_csc(AD.m, AD.m, 1);
        multiply(A_dense_csc, AT_dense_transpose, ADAT_dense_csc);

        dense_matrix_t<i_t, f_t> ADAT_dense(AD.m, AD.m);
        for (i_t k = 0; k < AD.m; k++) {
          ADAT_dense.from_sparse(ADAT_dense_csc, k, k);
        }

        f_t max_error = 0.0;
        for (i_t i = 0; i < AD.m; i++) {
          for (i_t j = 0; j < AD.m; j++) {
            f_t ij_error = std::abs(ADAT_dense(i, j) - UUT(i, j));
            max_error    = std::max(max_error, ij_error);
          }
        }

        settings_.log.printf("|| ADAT_dense - UUT ||_2 = %e\n", max_error);

        csc_matrix_t<i_t, f_t> A_sparse = A;
        std::vector<i_t> remove_dense(A.n, 0);
        for (i_t k : dense_columns) {
          remove_dense[k] = 1;
        }
        A_sparse.remove_columns(remove_dense);

        std::vector<f_t> inv_diag_sparse(A.n - n_dense_columns);
        i_t new_j = 0;
        for (i_t j = 0; j < A.n; j++) {
          if (cols_to_remove[j]) { continue; }
          inv_diag_sparse[new_j++] = std::sqrt(inv_diag[j]);
        }
        A_sparse.scale_columns(inv_diag_sparse);

        csc_matrix_t<i_t, f_t> AT_sparse_transpose(1, 1, 1);
        A_sparse.transpose(AT_sparse_transpose);

        csc_matrix_t<i_t, f_t> ADAT_sparse(AD.m, AD.m, 1);
        multiply(A_sparse, AT_sparse_transpose, ADAT_sparse);

        csc_matrix_t<i_t, f_t> error(AD.m, AD.m, 1);
        add(ADAT_sparse, ADAT, 1.0, -1.0, error);

        settings_.log.printf("|| ADAT_sparse - ADAT ||_1 = %e\n", error.norm1());

        csc_matrix_t<i_t, f_t> ADAT_test(AD.m, AD.m, 1);
        add(ADAT_sparse, ADAT_dense_csc, 1.0, 1.0, ADAT_test);

        csc_matrix_t<i_t, f_t> ADAT_all_columns(AD.m, AD.m, 1);
        csc_matrix_t<i_t, f_t> AT_all_columns(AD.n, AD.m, 1);
        A.transpose(AT_all_columns);
        csc_matrix_t<i_t, f_t> A_scaled = A;
        A_scaled.scale_columns(inv_diag);
        multiply(A_scaled, AT_all_columns, ADAT_all_columns);

        csc_matrix_t<i_t, f_t> error2(AD.m, AD.m, 1);
        add(ADAT_test, ADAT_all_columns, 1.0, -1.0, error2);

        int64_t large_nz = 0;
        for (i_t j = 0; j < AD.m; j++) {
          i_t col_start = error2.col_start[j];
          i_t col_end   = error2.col_start[j + 1];
          for (i_t p = col_start; p < col_end; p++) {
            if (std::abs(error2.x[p]) > 1e-6) {
              large_nz++;
              settings_.log.printf(
                "large_nz (%d,%d) %e. m %d\n", error2.i[p], j, error2.x[p], AD.m);
            }
          }
        }

        settings_.log.printf(
          "|| A_sparse * D_sparse * A_sparse^T + A_dense * D_dense * A_dense^T - ADAT ||_1 = %e "
          "nz "
          "%e large_nz %ld\n",
          error2.norm1(),
          static_cast<f_t>(error2.col_start[AD.m]),
          large_nz);
      }

      if (0 && debug) {
        f_t max_error     = 0.0;
        f_t max_row_error = 0.0;
        for (i_t i = 0; i < AD.m; i++) {
          dense_vector_t<i_t, f_t> ei(AD.m);
          ei.set_scalar(0.0);
          ei[i] = 1.0;

          dense_vector_t<i_t, f_t> VTei(n_dense_columns);
          AD_dense.transpose_multiply(1.0, ei, 0.0, VTei);

          f_t row_error = 0.0;
          for (i_t k = 0; k < n_dense_columns; k++) {
            i_t j = dense_columns[k];
            row_error += std::abs(VTei[k] - AD_dense(i, k));
          }
          if (row_error > 1e-10) { settings_.log.printf("row_error %d = %e\n", i, row_error); }
          max_row_error = std::max(max_row_error, row_error);

          dense_vector_t<i_t, f_t> u(AD.m);
          A_dense.matrix_vector_multiply(1.0, VTei, 0.0, u);

          matrix_vector_multiply(ADAT, 1.0, ei, 1.0, u);

          adat_multiply(-1.0, ei, 1.0, u);

          max_error = std::max(max_error, vector_norm2<i_t, f_t>(u));
        }
        settings_.log.printf(
          "|| (A_sparse * D_sparse * A_sparse^T + U * V^T) * e_i - ADA^T * e_i ||_2 = %e\n",
          max_error);
      }

      if (debug) {
        dense_vector_t<i_t, f_t> total_residual = b;
        adat_multiply(1.0, x, -1.0, total_residual);
        settings_.log.printf("|| A * D * A^T * x - b ||_2 = %e\n",
                             vector_norm2<i_t, f_t>(total_residual));
      }

      // Now do some rounds of PCG
      const bool do_pcg = true;
      if (do_pcg) { preconditioned_conjugate_gradient(settings_, b, 1e-9, x); }

      return solve_status;
    }
  }

  i_t gpu_solve_adat(rmm::device_uvector<f_t>& d_b, rmm::device_uvector<f_t>& d_x)
  {
    if (n_dense_columns == 0) {
      // Solve ADAT * x = b
      return chol->solve(d_b, d_x);
    } else {
      // TMP until this is ported to the GPU
      dense_vector_t<i_t, f_t> b = host_copy(d_b);
      dense_vector_t<i_t, f_t> x = host_copy(d_x);

      i_t out = solve_adat(b, x);

      d_b.resize(b.size(), stream_view_);
      d_x.resize(x.size(), stream_view_);
      raft::copy(d_b.data(), b.data(), b.size(), stream_view_);
      raft::copy(d_x.data(), x.data(), x.size(), stream_view_);

      return out;
    }
  }

  void to_solution(const lp_problem_t<i_t, f_t>& lp,
                   i_t iterations,
                   f_t objective,
                   f_t user_objective,
                   f_t primal_residual,
                   f_t dual_residual,
                   cusparse_view_t<i_t, f_t>& cusparse_view,
                   lp_solution_t<i_t, f_t>& solution)
  {
    solution.x = copy(x);
    solution.y = y;
    dense_vector_t<i_t, f_t> z_tilde(z.size());
    scatter_upper_bounds(v, z_tilde);
    z_tilde.axpy(1.0, z, -1.0);
    solution.z = z_tilde;

    dense_vector_t<i_t, f_t> dual_res = z_tilde;
    dual_res.axpy(-1.0, lp.objective, 1.0);
    if (use_gpu) {
      cusparse_view.transpose_spmv(1.0, solution.y, 1.0, dual_res);

    } else {
      matrix_transpose_vector_multiply(lp.A, 1.0, solution.y, 1.0, dual_res);
    }
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_res, stream_view_);
    settings_.log.printf("Solution Dual residual: %e\n", dual_residual_norm);

    solution.iterations         = iterations;
    solution.objective          = objective;
    solution.user_objective     = user_objective;
    solution.l2_primal_residual = primal_residual;
    solution.l2_dual_residual   = dual_residual;
  }

  void find_dense_columns(const csc_matrix_t<i_t, f_t>& A,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          std::vector<i_t>& columns_to_remove)
  {
    f_t start_column_density = tic();
    const i_t m              = A.m;
    const i_t n              = A.n;

    // Quick return if the problem is small
    if (m < 500) { return; }

    // The goal of this function is to find a set of dense columns in A
    // If a column of A is (partially) dense, it will cause A*A^T to be completely full.
    //
    // We can write A*A^T = sum_j A(:, j) * A(:, j)^T
    // We can split A*A^T into two parts
    // A*A^T =  sum_{j such that A(:, j) is sparse} A(:, j) * A(:, j)^T
    //        + sum_{j such that A(:, j) is dense} A(:, j) * A(:, j)^T
    // We call the first term A_sparse * A_sparse^T and the second term A_dense * A_dense^T
    //
    // We can then perform a sparse factorization of A_sparse * A_sparse^T
    // And use Schur complement techniques to extend this to allow us to solve with all of A*A^T

    // Thus, our goal is to find the columns that add the largest number of nonzeros to A*A^T
    // It is too expensive for us to compute the exact sparsity pattern that each column of A
    // contributes to A*A^T. Instead, we will use a heuristic method to estimate this.
    // This function roughly follows the approach taken in the paper:
    //
    //
    //  Meszaros, C. Detecting "dense" columns in interior point methods for linear programs.
    //  Comput Optim Appl 36, 309-320 (2007). https://doi.org/10.1007/s10589-006-9008-6
    //
    // But the reason for this detailed comment is to explain what the algorithm
    // given in the paper is doing.
    //
    // A loose upper bound is that column j contributes  |A(:, j) |^2 nonzeros to A*A^T
    // However, this upper bound assumes that each column of A is independent, when in
    // fact there is overlap in the sparsity pattern of A(:, j_1) and A(:, j_2)
    //
    //
    // Sort the columns of A according to their number of nonzeros
    std::vector<i_t> column_nz(n);
    i_t max_col_nz = 0;
    for (i_t j = 0; j < n; j++) {
      column_nz[j] = A.col_start[j + 1] - A.col_start[j];
      max_col_nz   = std::max(column_nz[j], max_col_nz);
    }
    if (max_col_nz < 100) { return; }  // Quick return if all columns of A have few nonzeros
    std::vector<i_t> column_nz_permutation(n);
    std::iota(column_nz_permutation.begin(), column_nz_permutation.end(), 0);
    std::sort(column_nz_permutation.begin(),
              column_nz_permutation.end(),
              [&column_nz](i_t i, i_t j) { return column_nz[i] < column_nz[j]; });

    // We then compute the exact sparsity pattern for columns of A whose where
    // the number of nonzeros is less than a threshold. This part can be done
    // quickly given that each column has only a few nonzeros. We will approximate
    // the effect of the dense columns a little later.

    const i_t threshold = 300;

    // Let C = A * A^T, the kth column of C is given by
    //
    // C(:, k) = A * A^T(:, k)
    //         = A * A(k, :)^T
    //         = sum_{j=1}^n A(:, j) * A(k, j)
    //         = sum_{j : A(k, j) != 0} A(:, j) * A(k, j)
    //
    // Thus we can compute the sparsity pattern associated with
    // the kth column of C by maintaining a single array of size m
    // and adding entries into that array as we traverse different
    // columns A(:, j)

    std::vector<i_t> mark(m, 0);

    // We will compute two arrays
    std::vector<i_t> column_count(m, 0);  // column_count[k] = number of nonzeros in C(:, k)
    std::vector<int64_t> delta_nz(n, 0);  // delta_nz[j] = additional fill in C due to A(:, j)

    // Note that we need to find j such that A(k, j) != 0.
    // The best way to do that is to have A stored in CSR format.
    csr_matrix_t<i_t, f_t> A_row(0, 0, 0);
    A.to_compressed_row(A_row);

    std::vector<i_t> histogram(m, 0);
    for (i_t j = 0; j < n; j++) {
      const i_t col_nz_j = A.col_start[j + 1] - A.col_start[j];
      histogram[col_nz_j]++;
    }
    settings.log.printf("Col Nz  # cols\n");
    for (i_t k = 0; k < m; k++) {
      if (histogram[k] > 0) { settings.log.printf("%6d %6d\n", k, histogram[k]); }
    }
    settings.log.printf("\n");

    std::vector<i_t> row_nz(m, 0);
    for (i_t j = 0; j < n; j++) {
      const i_t col_start = A.col_start[j];
      const i_t col_end   = A.col_start[j + 1];
      for (i_t p = col_start; p < col_end; p++) {
        row_nz[A.i[p]]++;
      }
    }

    std::vector<i_t> histogram_row(m, 0);
    for (i_t k = 0; k < m; k++) {
      histogram_row[row_nz[k]]++;
    }
    settings.log.printf("Row Nz  # rows\n");
    for (i_t k = 0; k < m; k++) {
      if (histogram_row[k] > 0) { settings.log.printf("%6d %6d\n", k, histogram_row[k]); }
    }

    for (i_t k = 0; k < m; k++) {
      // The nonzero pattern of C(:, k) will be those entries with mark[i] = k
      const i_t row_start = A_row.row_start[k];
      const i_t row_end   = A_row.row_start[k + 1];
      for (i_t p = row_start; p < row_end; p++) {
        const i_t j = A_row.j[p];
        int64_t fill =
          0;  // This will hold the additional fill coming from A(:, j) in the current pass
        const i_t col_start = A.col_start[j];
        const i_t col_end   = A.col_start[j + 1];
        const i_t col_nz_j  = col_end - col_start;
        // settings.log.printf("col_nz_j %6d j %6d\n", col_nz_j, j);
        if (col_nz_j > threshold) { continue; }  // Skip columns above the threshold
        for (i_t q = col_start; q < col_end; q++) {
          const i_t i = A.i[q];
          // settings.log.printf("A(%d, %d) i %6d mark[%d] = %6d =? %6d\n", i, j, i, i, mark[i],
          // k);
          if (mark[i] != k) {  // We have generated some fill in C(:, k)
            mark[i] = k;
            fill++;
            // settings.log.printf("Fill %6d %6d\n", k, i);
          }
        }
        column_count[k] += fill;  // Add in the contributions from A(:, j) to C(:, k). Since fill
                                  // will be zeroed at next iteration.
        delta_nz[j] +=
          fill;  // Capture contributions from A(:, j). j will be encountered multiple times
      }
    }

    int64_t sparse_nz_C = 0;
    for (i_t j = 0; j < n; j++) {
      sparse_nz_C += delta_nz[j];
    }
    settings.log.printf("Sparse nz AAT %e\n", static_cast<f_t>(sparse_nz_C));

    // Now we estimate the fill in C due to the dense columns
    i_t num_estimated_columns = 0;
    for (i_t k = 0; k < n; k++) {
      const i_t j = column_nz_permutation[k];  // We want to traverse columns in order of
                                               // increasing number of nonzeros
      const i_t col_nz_j = A.col_start[j + 1] - A.col_start[j];
      if (col_nz_j <= threshold) { continue; }
      num_estimated_columns++;
      // This column will contribute A(:, j) * A(: j)^T to C
      // The columns of C that will be affected are k such that A(k, j) ! = 0
      const i_t col_start = A.col_start[j];
      const i_t col_end   = A.col_start[j + 1];
      for (i_t q = col_start; q < col_end; q++) {
        const i_t k = A.i[q];
        // The max possible fill in C(:, k) is | A(:, j) |
        f_t max_possible = static_cast<f_t>(col_nz_j);
        // But if the C(:, k) = m, i.e the column is already full, there will be no fill.
        // So we use the following heuristic
        const f_t fraction_filled = 1.0 * static_cast<f_t>(column_count[k]) / static_cast<f_t>(m);
        f_t fill_estimate         = max_possible * (1.0 - fraction_filled);
        column_count[k] =
          std::min(m,
                   column_count[k] +
                     static_cast<i_t>(fill_estimate));  // Capture the estimated fill to C(:, k)
        delta_nz[j] = std::min(
          static_cast<int64_t>(m) * static_cast<int64_t>(m),
          delta_nz[j] + static_cast<int64_t>(
                          fill_estimate));  // Capture the estimated fill associated with column j
      }
    }

    int64_t estimated_nz_C = 0;
    for (i_t i = 0; i < m; i++) {
      estimated_nz_C += static_cast<int64_t>(column_count[i]);
    }
    settings.log.printf("Estimated nz AAT %e\n", static_cast<f_t>(estimated_nz_C));

    // Sort the columns of A according to their additional fill
    std::vector<i_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&delta_nz](i_t i, i_t j) {
      return delta_nz[i] < delta_nz[j];
    });

    // Now we make a forward pass and compute the number of nonzeros in C
    // assuming we had included column j
    std::vector<f_t> cumulative_nonzeros(n, 0.0);
    int64_t nnz_C = 0;
    for (i_t k = 0; k < n; k++) {
      const i_t j = permutation[k];
      // settings.log.printf("Column %6d delta nz %d\n", j, delta_nz[j]);
      nnz_C += delta_nz[j];
      cumulative_nonzeros[k] = static_cast<f_t>(nnz_C);
      if (n - k < 10) {
        settings.log.printf("Cumulative nonzeros %ld %6.2e k %6d delta nz %ld col %6d\n",
                            nnz_C,
                            cumulative_nonzeros[k],
                            k,
                            delta_nz[j],
                            j);
      }
    }
    settings.log.printf("Cumulative nonzeros %ld %6.2e\n", nnz_C, cumulative_nonzeros[n - 1]);

    // Forward pass again to pick up the dense columns
    columns_to_remove.reserve(n);
    f_t total_nz_estimate = cumulative_nonzeros[n - 1];
    for (i_t k = 1; k < n; k++) {
      const i_t j     = permutation[k];
      i_t col_nz      = A.col_start[j + 1] - A.col_start[j];
      f_t delta_nz_j  = std::max(static_cast<f_t>(col_nz * col_nz),
                                cumulative_nonzeros[k] - cumulative_nonzeros[k - 1]);
      const f_t ratio = delta_nz_j / total_nz_estimate;
      if (ratio > .01) {
        settings.log.printf(
          "Column: nz %10d cumulative nz %6.2e estimated delta nz %6.2e percent %.2f col %6d\n",
          col_nz,
          cumulative_nonzeros[k],
          delta_nz_j,
          ratio,
          j);
        columns_to_remove.push_back(j);
      }
    }
  }

  template <typename AllocatorA, typename AllocatorB>
  void scatter_upper_bounds(const dense_vector_t<i_t, f_t, AllocatorA>& y,
                            dense_vector_t<i_t, f_t, AllocatorB>& z)
  {
    if (n_upper_bounds > 0) {
      for (i_t k = 0; k < n_upper_bounds; k++) {
        i_t j = upper_bounds[k];
        z[j]  = y[k];
      }
    }
  }

  template <typename AllocatorA, typename AllocatorB>
  void gather_upper_bounds(const std::vector<f_t, AllocatorA>& z, std::vector<f_t, AllocatorB>& y)
  {
    if (n_upper_bounds > 0) {
      for (i_t k = 0; k < n_upper_bounds; k++) {
        i_t j = upper_bounds[k];
        y[k]  = z[j];
      }
    }
  }

  // v = alpha * A * Dinv * A^T * y + beta * v
  void gpu_adat_multiply(f_t alpha,
                         const rmm::device_uvector<f_t>& y,
                         cusparseDnVecDescr_t cusparse_y,
                         f_t beta,
                         rmm::device_uvector<f_t>& v,
                         cusparseDnVecDescr_t cusparse_v,
                         rmm::device_uvector<f_t>& u,
                         cusparseDnVecDescr_t cusparse_u,
                         cusparse_view_t<i_t, f_t>& cusparse_view,
                         const rmm::device_uvector<f_t>& d_inv_diag) const
  {
    raft::common::nvtx::range fun_scope("Barrier: gpu_adat_multiply");

    const i_t m = A.m;
    const i_t n = A.n;

    if (y.size() != m) {
      printf("adat_multiply: y.size() %d != m %d\n", static_cast<i_t>(y.size()), m);
      exit(1);
    }

    if (v.size() != m) {
      printf("adat_multiply: v.size() %d != m %d\n", static_cast<i_t>(v.size()), m);
      exit(1);
    }

    // v = alpha * A * Dinv * A^T * y + beta * v

    // u = A^T * y

    cusparse_view.transpose_spmv(1.0, cusparse_y, 0.0, cusparse_u);

    // u = Dinv * u
    cub::DeviceTransform::Transform(cuda::std::make_tuple(u.data(), d_inv_diag.data()),
                                    u.data(),
                                    u.size(),
                                    cuda::std::multiplies<>{},
                                    stream_view_);

    // y = alpha * A * w + beta * v = alpha * A * Dinv * A^T * y + beta * v
    cusparse_view.spmv(alpha, cusparse_u, beta, cusparse_v);
  }

  // v = alpha * A * Dinv * A^T * y + beta * v
  void adat_multiply(f_t alpha,
                     const dense_vector_t<i_t, f_t>& y,
                     f_t beta,
                     dense_vector_t<i_t, f_t>& v) const
  {
    const i_t m = A.m;
    const i_t n = A.n;

    if (static_cast<i_t>(y.size()) != m) {
      printf("adat_multiply: y.size() %d != m %d\n", static_cast<i_t>(y.size()), m);
      exit(1);
    }

    if (static_cast<i_t>(v.size()) != m) {
      printf("adat_multiply: v.size() %d != m %d\n", static_cast<i_t>(v.size()), m);
      exit(1);
    }

    // v = alpha * A * Dinv * A^T * y + beta * v

    // u = A^T * y
    dense_vector_t<i_t, f_t> u(n);
    matrix_transpose_vector_multiply(A, 1.0, y, 0.0, u);

    // w = Dinv * u
    dense_vector_t<i_t, f_t> w(n);
    inv_diag.pairwise_product(u, w);

    // v = alpha * A * w + beta * v = alpha * A * Dinv * A^T * y + beta * v
    matrix_vector_multiply(A, alpha, w, beta, v);
  }

  void preconditioned_conjugate_gradient(const simplex_solver_settings_t<i_t, f_t>& settings,
                                         const dense_vector_t<i_t, f_t>& b,
                                         f_t tolerance,
                                         dense_vector_t<i_t, f_t>& xinout) const
  {
    const bool show_pcg_info          = false;
    dense_vector_t<i_t, f_t> residual = b;
    dense_vector_t<i_t, f_t> y(ADAT.n);

    dense_vector_t<i_t, f_t> x = xinout;

    const csc_matrix_t<i_t, f_t>& A = ADAT;

    // r = A*x - b
    adat_multiply(1.0, x, -1.0, residual);

    // Solve M y = r for y
    chol->solve(residual, y);

    dense_vector_t<i_t, f_t> p = y;
    p.multiply_scalar(-1.0);

    dense_vector_t<i_t, f_t> Ap(A.n);
    i_t iter                  = 0;
    f_t norm_residual         = vector_norm2<i_t, f_t>(residual);
    f_t initial_norm_residual = norm_residual;
    if (show_pcg_info) {
      settings.log.printf("PCG initial residual 2-norm %e inf-norm %e\n",
                          norm_residual,
                          vector_norm_inf<i_t, f_t>(residual, stream_view_));
    }

    f_t rTy = residual.inner_product(y);

    while (norm_residual > tolerance && iter < 100) {
      // Compute Ap = A * p
      adat_multiply(1.0, p, 0.0, Ap);

      // Compute alpha = (r^T * y) / (p^T * Ap)
      f_t alpha = rTy / p.inner_product(Ap);

      // Update residual = residual + alpha * Ap
      residual.axpy(alpha, Ap, 1.0);

      f_t new_residual = vector_norm2<i_t, f_t>(residual);
      if (new_residual > 1.1 * norm_residual || new_residual > 1.1 * initial_norm_residual) {
        if (show_pcg_info) {
          settings.log.printf(
            "PCG residual increased by more than 10%%. New %e > %e\n", new_residual, norm_residual);
        }
        break;
      }
      norm_residual = new_residual;

      // Update x = x + alpha * p
      x.axpy(alpha, p, 1.0);

      // residual = A*x - b
      residual = b;
      adat_multiply(1.0, x, -1.0, residual);
      norm_residual = vector_norm2<i_t, f_t>(residual);

      // Solve M y = r for y
      chol->solve(residual, y);

      // Compute beta = (r_+^T y_+) / (r^T y)
      f_t r1Ty1 = residual.inner_product(y);
      f_t beta  = r1Ty1 / rTy;

      rTy = r1Ty1;

      // Update p = -y + beta * p
      p.axpy(-1.0, y, beta);

      iter++;

      if (show_pcg_info) {
        settings.log.printf("PCG iter %3d 2-norm_residual %.2e inf-norm_residual %.2e\n",
                            iter,
                            norm_residual,
                            vector_norm_inf<i_t, f_t>(residual, stream_view_));
      }
    }

    residual = b;
    adat_multiply(1.0, x, -1.0, residual);
    norm_residual = vector_norm2<i_t, f_t>(residual);
    if (norm_residual < initial_norm_residual) {
      if (show_pcg_info) {
        settings.log.printf("PCG improved residual 2-norm %.2e/%.2e in %d iterations\n",
                            norm_residual,
                            initial_norm_residual,
                            iter);
      }
      xinout = x;
    } else {
      if (show_pcg_info) { settings.log.printf("Rejecting PCG solution\n"); }
    }
  }

  raft::handle_t const* handle_ptr;
  i_t n_upper_bounds;
  pinned_dense_vector_t<i_t, i_t> upper_bounds;
  pinned_dense_vector_t<i_t, f_t> c;
  pinned_dense_vector_t<i_t, f_t> b;

  pinned_dense_vector_t<i_t, f_t> w;
  pinned_dense_vector_t<i_t, f_t> x;
  dense_vector_t<i_t, f_t> y;
  pinned_dense_vector_t<i_t, f_t> v;
  pinned_dense_vector_t<i_t, f_t> z;

  dense_vector_t<i_t, f_t> w_save;
  dense_vector_t<i_t, f_t> x_save;
  dense_vector_t<i_t, f_t> y_save;
  dense_vector_t<i_t, f_t> v_save;
  dense_vector_t<i_t, f_t> z_save;

  pinned_dense_vector_t<i_t, f_t> diag;
  pinned_dense_vector_t<i_t, f_t> inv_diag;
  pinned_dense_vector_t<i_t, f_t> inv_sqrt_diag;

  std::vector<f_t> original_A_values;
  rmm::device_uvector<f_t> d_original_A_values;

  csc_matrix_t<i_t, f_t> AD;
  csc_matrix_t<i_t, f_t> AT;
  csc_matrix_t<i_t, f_t> ADAT;
  device_csr_matrix_t<i_t, f_t> device_ADAT;
  device_csr_matrix_t<i_t, f_t> device_A;
  device_csc_matrix_t<i_t, f_t> device_AD;
  rmm::device_uvector<f_t> device_A_x_values;
  // For GPU Form ADAT
  rmm::device_uvector<f_t> d_inv_diag_prime;
  rmm::device_buffer d_flag_buffer;
  size_t flag_buffer_size;
  rmm::device_scalar<i_t> d_num_flag;
  rmm::device_uvector<f_t> d_inv_diag;

  i_t n_dense_columns;
  std::vector<i_t> dense_columns;
  std::vector<i_t> sparse_mark;
  std::vector<i_t> cols_to_remove;
  rmm::device_uvector<i_t> d_cols_to_remove;
  dense_matrix_t<i_t, f_t> A_dense;
  dense_matrix_t<i_t, f_t> AD_dense;
  dense_matrix_t<i_t, f_t> H;
  dense_matrix_t<i_t, f_t> Hchol;
  const csc_matrix_t<i_t, f_t>& A;

  std::unique_ptr<sparse_cholesky_base_t<i_t, f_t>> chol;

  bool has_factorization;
  bool has_solve_info;
  i_t num_factorizations;

  pinned_dense_vector_t<i_t, f_t> primal_residual;
  pinned_dense_vector_t<i_t, f_t> bound_residual;
  pinned_dense_vector_t<i_t, f_t> dual_residual;
  pinned_dense_vector_t<i_t, f_t> complementarity_xz_residual;
  pinned_dense_vector_t<i_t, f_t> complementarity_wv_residual;

  pinned_dense_vector_t<i_t, f_t> primal_rhs;
  pinned_dense_vector_t<i_t, f_t> bound_rhs;
  pinned_dense_vector_t<i_t, f_t> dual_rhs;
  pinned_dense_vector_t<i_t, f_t> complementarity_xz_rhs;
  pinned_dense_vector_t<i_t, f_t> complementarity_wv_rhs;

  pinned_dense_vector_t<i_t, f_t> dw_aff;
  pinned_dense_vector_t<i_t, f_t> dx_aff;
  pinned_dense_vector_t<i_t, f_t> dy_aff;
  pinned_dense_vector_t<i_t, f_t> dv_aff;
  pinned_dense_vector_t<i_t, f_t> dz_aff;

  pinned_dense_vector_t<i_t, f_t> dw;
  pinned_dense_vector_t<i_t, f_t> dx;
  pinned_dense_vector_t<i_t, f_t> dy;
  pinned_dense_vector_t<i_t, f_t> dv;
  pinned_dense_vector_t<i_t, f_t> dz;
  cusparse_info_t<i_t, f_t> cusparse_info;
  cusparse_view_t<i_t, f_t> cusparse_view_;
  cusparseDnVecDescr_t cusparse_tmp4_;
  cusparseDnVecDescr_t cusparse_h_;
  cusparseDnVecDescr_t cusparse_dx_residual_;
  cusparseDnVecDescr_t cusparse_dy_;
  cusparseDnVecDescr_t cusparse_dx_residual_5_;
  cusparseDnVecDescr_t cusparse_dx_residual_6_;
  cusparseDnVecDescr_t cusparse_dx_;
  cusparseDnVecDescr_t cusparse_dx_residual_3_;
  cusparseDnVecDescr_t cusparse_dx_residual_4_;
  cusparseDnVecDescr_t cusparse_r1_;
  cusparseDnVecDescr_t cusparse_dual_residual_;
  cusparseDnVecDescr_t cusparse_y_residual_;
  // GPU ADAT multiply
  cusparseDnVecDescr_t cusparse_u_;

  transform_reduce_helper_t<f_t> transform_reduce_helper_;
  sum_reduce_helper_t<f_t> sum_reduce_helper_;

  rmm::cuda_stream_view stream_view_;

  const simplex_solver_settings_t<i_t, f_t>& settings_;
};

template <typename i_t, typename f_t>
barrier_solver_t<i_t, f_t>::barrier_solver_t(const lp_problem_t<i_t, f_t>& lp,
                                             const presolve_info_t<i_t, f_t>& presolve,
                                             const simplex_solver_settings_t<i_t, f_t>& settings)
  : lp(lp),
    settings(settings),
    presolve_info(presolve),
    stream_view_(lp.handle_ptr->get_stream()),
    d_diag_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_bound_rhs_(0, lp.handle_ptr->get_stream()),
    d_x_(0, lp.handle_ptr->get_stream()),
    d_z_(0, lp.handle_ptr->get_stream()),
    d_w_(0, lp.handle_ptr->get_stream()),
    d_v_(0, lp.handle_ptr->get_stream()),
    d_h_(lp.num_rows, lp.handle_ptr->get_stream()),
    d_y_(0, lp.handle_ptr->get_stream()),
    d_upper_bounds_(0, lp.handle_ptr->get_stream()),
    d_tmp3_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_tmp4_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_r1_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_r1_prime_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_dy_(0, lp.handle_ptr->get_stream()),
    d_c_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_dx_(0, lp.handle_ptr->get_stream()),
    d_dual_residual_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_complementarity_xz_rhs_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_complementarity_wv_rhs_(0, lp.handle_ptr->get_stream()),
    d_dual_rhs_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_dz_(0, lp.handle_ptr->get_stream()),
    d_dv_(0, lp.handle_ptr->get_stream()),
    d_y_residual_(lp.num_rows, lp.handle_ptr->get_stream()),
    d_dx_residual_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_xz_residual_(0, lp.handle_ptr->get_stream()),
    d_dw_(0, lp.handle_ptr->get_stream()),
    d_dw_residual_(0, lp.handle_ptr->get_stream()),
    d_wv_residual_(0, lp.handle_ptr->get_stream()),
    d_dw_aff_(0, lp.handle_ptr->get_stream()),
    d_dx_aff_(0, lp.handle_ptr->get_stream()),
    d_dv_aff_(0, lp.handle_ptr->get_stream()),
    d_dz_aff_(0, lp.handle_ptr->get_stream()),
    d_dy_aff_(0, lp.handle_ptr->get_stream()),
    d_u_(lp.A.n, lp.handle_ptr->get_stream()),
    d_primal_residual_(0, lp.handle_ptr->get_stream()),
    d_bound_residual_(0, lp.handle_ptr->get_stream()),
    d_upper_(0, lp.handle_ptr->get_stream()),
    d_complementarity_xz_residual_(lp.num_cols, lp.handle_ptr->get_stream()),
    d_complementarity_wv_residual_(0, lp.handle_ptr->get_stream()),
    restrict_u_(0)
{
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::initial_point(iteration_data_t<i_t, f_t>& data)
{
  raft::common::nvtx::range fun_scope("Barrier: initial_point");

  // Perform a numerical factorization
  i_t status;
  if (use_gpu) {
    status = data.chol->factorize(data.device_ADAT);
  } else {
    status = data.chol->factorize(data.ADAT);
  }
  if (status != 0) {
    settings.log.printf("Initial factorization failed\n");
    return;
  }
  data.num_factorizations++;
  data.has_solve_info = false;

  // rhs_x <- b
  dense_vector_t<i_t, f_t> rhs_x(lp.rhs);

  dense_vector_t<i_t, f_t> Fu(lp.num_cols);
  data.gather_upper_bounds(lp.upper, Fu);

  dense_vector_t<i_t, f_t> DinvFu(lp.num_cols);  // DinvFu = Dinv * Fu
  data.inv_diag.pairwise_product(Fu, DinvFu);

  // rhs_x <-  A * Dinv * F * u  - b
  if (use_gpu) {
    data.cusparse_view_.spmv(1.0, DinvFu, -1.0, rhs_x);
  } else {
    matrix_vector_multiply(lp.A, 1.0, DinvFu, -1.0, rhs_x);
  }

  // Solve A*Dinv*A'*q = A*Dinv*F*u - b
  dense_vector_t<i_t, f_t> q(lp.num_rows);
  settings.log.printf("||rhs_x|| = %e\n", vector_norm2<i_t, f_t>(rhs_x));
  // i_t solve_status = data.chol->solve(rhs_x, q);
  i_t solve_status = data.solve_adat(rhs_x, q);
  settings.log.printf("Initial solve status %d\n", solve_status);
  settings.log.printf("||q|| = %e\n", vector_norm2<i_t, f_t>(q));

  // rhs_x <- A*Dinv*A'*q - rhs_x
  data.adat_multiply(1.0, q, -1.0, rhs_x);
  // matrix_vector_multiply(data.ADAT, 1.0, q, -1.0, rhs_x);
  settings.log.printf("|| A*Dinv*A'*q - (A*Dinv*F*u - b) || = %e\n", vector_norm2<i_t, f_t>(rhs_x));

  // x = Dinv*(F*u - A'*q)
  data.inv_diag.pairwise_product(Fu, data.x);
  // Fu <- -1.0 * A' * q + 1.0 * Fu
  if (use_gpu) {
    data.cusparse_view_.transpose_spmv(-1.0, q, 1.0, Fu);
  } else {
    matrix_transpose_vector_multiply(lp.A, -1.0, q, 1.0, Fu);
  }

  // x <- Dinv * (F*u - A'*q)
  data.inv_diag.pairwise_product(Fu, data.x);

  // w <- E'*u - E'*x
  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j     = data.upper_bounds[k];
      data.w[k] = lp.upper[j] - data.x[j];
    }
  }

  // Verify A*x = b
  data.primal_residual = lp.rhs;
  if (use_gpu) {
    data.cusparse_view_.spmv(1.0, data.x, -1.0, data.primal_residual);
  } else {
    matrix_vector_multiply(lp.A, 1.0, data.x, -1.0, data.primal_residual);
  }
  settings.log.printf("||b - A * x||: %e\n", vector_norm2<i_t, f_t>(data.primal_residual));

  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j                  = data.upper_bounds[k];
      data.bound_residual[k] = lp.upper[j] - data.w[k] - data.x[j];
    }
    settings.log.printf("|| u - w - x||: %e\n", vector_norm2<i_t, f_t>(data.bound_residual));
  }

  // First compute rhs = A*Dinv*c
  dense_vector_t<i_t, f_t> rhs(lp.num_rows);
  dense_vector_t<i_t, f_t> Dinvc(lp.num_cols);
  data.inv_diag.pairwise_product(lp.objective, Dinvc);
  // rhs = 1.0 * A * Dinv * c
  if (use_gpu) {
    data.cusparse_view_.spmv(1.0, Dinvc, 0.0, rhs);
  } else {
    matrix_vector_multiply(lp.A, 1.0, Dinvc, 0.0, rhs);
  }

  // Solve A*Dinv*A'*q = A*Dinv*c
  // data.chol->solve(rhs, data.y);
  data.solve_adat(rhs, data.y);

  // z = Dinv*(c - A'*y)
  dense_vector_t<i_t, f_t> cmATy = data.c;
  if (use_gpu) {
    data.cusparse_view_.transpose_spmv(-1.0, data.y, 1.0, cmATy);
  } else {
    matrix_transpose_vector_multiply(lp.A, -1.0, data.y, 1.0, cmATy);
  }
  // z <- Dinv * (c - A'*y)
  data.inv_diag.pairwise_product(cmATy, data.z);

  // v = -E'*z
  data.gather_upper_bounds(data.z, data.v);
  data.v.multiply_scalar(-1.0);

  // Verify A'*y + z - E*v = c
  data.z.pairwise_subtract(data.c, data.dual_residual);
  if (use_gpu) {
    data.cusparse_view_.transpose_spmv(1.0, data.y, 1.0, data.dual_residual);
  } else {
    matrix_transpose_vector_multiply(lp.A, 1.0, data.y, 1.0, data.dual_residual);
  }
  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j = data.upper_bounds[k];
      data.dual_residual[j] -= data.v[k];
    }
  }
  settings.log.printf("||A^T y + z - E*v - c ||: %e\n", vector_norm2<i_t, f_t>(data.dual_residual));

  // Make sure (w, x, v, z) > 0
  float64_t epsilon_adjust = 10.0;
  data.w.ensure_positive(epsilon_adjust);
  data.x.ensure_positive(epsilon_adjust);
  data.v.ensure_positive(epsilon_adjust);
  data.z.ensure_positive(epsilon_adjust);
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::gpu_compute_residuals(const rmm::device_uvector<f_t>& d_w,
                                                       const rmm::device_uvector<f_t>& d_x,
                                                       const rmm::device_uvector<f_t>& d_y,
                                                       const rmm::device_uvector<f_t>& d_v,
                                                       const rmm::device_uvector<f_t>& d_z,
                                                       iteration_data_t<i_t, f_t>& data)
{
  raft::common::nvtx::range fun_scope("Barrier: GPU compute_residuals");

  d_primal_residual_.resize(data.primal_residual.size(), stream_view_);
  raft::copy(d_primal_residual_.data(), lp.rhs.data(), lp.rhs.size(), stream_view_);

  d_dual_residual_.resize(data.dual_residual.size(), stream_view_);
  raft::copy(
    d_dual_residual_.data(), data.dual_residual.data(), data.dual_residual.size(), stream_view_);
  d_upper_bounds_.resize(data.n_upper_bounds, stream_view_);
  raft::copy(d_upper_bounds_.data(), data.upper_bounds.data(), data.n_upper_bounds, stream_view_);
  d_upper_.resize(lp.upper.size(), stream_view_);
  raft::copy(d_upper_.data(), lp.upper.data(), lp.upper.size(), stream_view_);
  d_bound_residual_.resize(data.bound_residual.size(), stream_view_);
  raft::copy(
    d_bound_residual_.data(), data.bound_residual.data(), data.bound_residual.size(), stream_view_);

  // Compute primal_residual = b - A*x

  auto cusparse_d_x          = data.cusparse_view_.create_vector(d_x);
  auto descr_primal_residual = data.cusparse_view_.create_vector(d_primal_residual_);
  data.cusparse_view_.spmv(-1.0, cusparse_d_x, 1.0, descr_primal_residual);

  // Compute bound_residual = E'*u - w - E'*x
  if (data.n_upper_bounds > 0) {
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        thrust::make_permutation_iterator(d_upper_.data(), d_upper_bounds_.data()),
        d_w.data(),
        thrust::make_permutation_iterator(d_x.data(), d_upper_bounds_.data())),
      d_bound_residual_.data(),
      d_upper_bounds_.size(),
      [] HD(f_t upper_j, f_t w_k, f_t x_j) { return upper_j - w_k - x_j; },
      stream_view_);
  }

  // Compute dual_residual = c - A'*y - z + E*v
  raft::copy(d_c_.data(), data.c.data(), data.c.size(), stream_view_);
  cub::DeviceTransform::Transform(cuda::std::make_tuple(d_c_.data(), d_z.data()),
                                  d_dual_residual_.data(),
                                  d_dual_residual_.size(),
                                  cuda::std::minus<>{},
                                  stream_view_);

  // Compute dual_residual = c - A'*y - z + E*v
  auto cusparse_d_y        = data.cusparse_view_.create_vector(d_y);
  auto descr_dual_residual = data.cusparse_view_.create_vector(d_dual_residual_);
  data.cusparse_view_.transpose_spmv(-1.0, cusparse_d_y, 1.0, descr_dual_residual);

  if (data.n_upper_bounds > 0) {
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        thrust::make_permutation_iterator(d_dual_residual_.data(), d_upper_bounds_.data()),
        d_v.data()),
      thrust::make_permutation_iterator(d_dual_residual_.data(), d_upper_bounds_.data()),
      d_upper_bounds_.size(),
      [] HD(f_t dual_residual_j, f_t v_k) { return dual_residual_j + v_k; },
      stream_view_);
  }

  // Compute complementarity_xz_residual = x.*z
  cub::DeviceTransform::Transform(cuda::std::make_tuple(d_x.data(), d_z.data()),
                                  d_complementarity_xz_residual_.data(),
                                  d_complementarity_xz_residual_.size(),
                                  cuda::std::multiplies<>{},
                                  stream_view_);

  // Compute complementarity_wv_residual = w.*v
  cub::DeviceTransform::Transform(cuda::std::make_tuple(d_w.data(), d_v.data()),
                                  d_complementarity_wv_residual_.data(),
                                  d_complementarity_wv_residual_.size(),
                                  cuda::std::multiplies<>{},
                                  stream_view_);
  raft::copy(data.complementarity_wv_residual.data(),
             d_complementarity_wv_residual_.data(),
             d_complementarity_wv_residual_.size(),
             stream_view_);
  raft::copy(data.complementarity_xz_residual.data(),
             d_complementarity_xz_residual_.data(),
             d_complementarity_xz_residual_.size(),
             stream_view_);
  raft::copy(
    data.dual_residual.data(), d_dual_residual_.data(), d_dual_residual_.size(), stream_view_);
  raft::copy(data.primal_residual.data(),
             d_primal_residual_.data(),
             d_primal_residual_.size(),
             stream_view_);
  raft::copy(
    data.bound_residual.data(), d_bound_residual_.data(), d_bound_residual_.size(), stream_view_);
  // Sync to make sure host data has been copied
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
}

template <typename i_t, typename f_t>
template <typename AllocatorA>
void barrier_solver_t<i_t, f_t>::compute_residuals(const dense_vector_t<i_t, f_t, AllocatorA>& w,
                                                   const dense_vector_t<i_t, f_t, AllocatorA>& x,
                                                   const dense_vector_t<i_t, f_t, AllocatorA>& y,
                                                   const dense_vector_t<i_t, f_t, AllocatorA>& v,
                                                   const dense_vector_t<i_t, f_t, AllocatorA>& z,
                                                   iteration_data_t<i_t, f_t>& data)
{
  raft::common::nvtx::range fun_scope("Barrier: CPU compute_residuals");

  // Compute primal_residual = b - A*x
  data.primal_residual = lp.rhs;
  if (use_gpu) {
    data.cusparse_view_.spmv(-1.0, x, 1.0, data.primal_residual);
  } else {
    matrix_vector_multiply(lp.A, -1.0, x, 1.0, data.primal_residual);
  }

  // Compute bound_residual = E'*u - w - E'*x
  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j                  = data.upper_bounds[k];
      data.bound_residual[k] = lp.upper[j] - w[k] - x[j];
    }
  }

  // Compute dual_residual = c - A'*y - z + E*v
  data.c.pairwise_subtract(z, data.dual_residual);
  if (use_gpu) {
    data.cusparse_view_.transpose_spmv(-1.0, y, 1.0, data.dual_residual);
  } else {
    matrix_transpose_vector_multiply(lp.A, -1.0, y, 1.0, data.dual_residual);
  }
  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j = data.upper_bounds[k];
      data.dual_residual[j] += v[k];
    }
  }

  // Compute complementarity_xz_residual = x.*z
  x.pairwise_product(z, data.complementarity_xz_residual);

  // Compute complementarity_wv_residual = w.*v
  w.pairwise_product(v, data.complementarity_wv_residual);
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::gpu_compute_residual_norms(const rmm::device_uvector<f_t>& d_w,
                                                            const rmm::device_uvector<f_t>& d_x,
                                                            const rmm::device_uvector<f_t>& d_y,
                                                            const rmm::device_uvector<f_t>& d_v,
                                                            const rmm::device_uvector<f_t>& d_z,
                                                            iteration_data_t<i_t, f_t>& data,
                                                            f_t& primal_residual_norm,
                                                            f_t& dual_residual_norm,
                                                            f_t& complementarity_residual_norm)
{
  raft::common::nvtx::range fun_scope("Barrier: GPU compute_residual_norms");

  gpu_compute_residuals(d_w, d_x, d_y, d_v, d_z, data);
  primal_residual_norm =
    std::max(device_vector_norm_inf<i_t, f_t>(d_primal_residual_, stream_view_),
             device_vector_norm_inf<i_t, f_t>(d_bound_residual_, stream_view_));
  dual_residual_norm = device_vector_norm_inf<i_t, f_t>(d_dual_residual_, stream_view_);
  complementarity_residual_norm =
    std::max(device_vector_norm_inf<i_t, f_t>(d_complementarity_xz_rhs_, stream_view_),
             device_vector_norm_inf<i_t, f_t>(d_complementarity_wv_rhs_, stream_view_));
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::compute_residual_norms(const dense_vector_t<i_t, f_t>& w,
                                                        const dense_vector_t<i_t, f_t>& x,
                                                        const dense_vector_t<i_t, f_t>& y,
                                                        const dense_vector_t<i_t, f_t>& v,
                                                        const dense_vector_t<i_t, f_t>& z,
                                                        iteration_data_t<i_t, f_t>& data,
                                                        f_t& primal_residual_norm,
                                                        f_t& dual_residual_norm,
                                                        f_t& complementarity_residual_norm)
{
  raft::common::nvtx::range fun_scope("Barrier: CPU compute_residual_norms");

  compute_residuals(w, x, y, v, z, data);
  primal_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.primal_residual, stream_view_),
                                  vector_norm_inf<i_t, f_t>(data.bound_residual, stream_view_));
  dual_residual_norm   = vector_norm_inf<i_t, f_t>(data.dual_residual, stream_view_);
  complementarity_residual_norm =
    std::max(vector_norm_inf<i_t, f_t>(data.complementarity_xz_residual, stream_view_),
             vector_norm_inf<i_t, f_t>(data.complementarity_wv_residual, stream_view_));
}

template <typename i_t, typename f_t>
template <typename AllocatorA, typename AllocatorB>
f_t barrier_solver_t<i_t, f_t>::max_step_to_boundary(const dense_vector_t<i_t, f_t, AllocatorA>& x,
                                                     const dense_vector_t<i_t, f_t, AllocatorB>& dx,
                                                     i_t& index) const
{
  float64_t max_step = 1.0;
  index              = -1;
  for (i_t i = 0; i < static_cast<i_t>(x.size()); i++) {
    // x_i + alpha * dx_i >= 0, x_i >= 0, alpha >= 0
    // We only need to worry about the case where dx_i < 0
    // alpha * dx_i >= -x_i => alpha <= -x_i / dx_i
    if (dx[i] < 0.0) {
      const f_t ratio = -x[i] / dx[i];
      if (ratio < max_step) {
        max_step = ratio;
        index    = i;
      }
    }
  }
  return max_step;
}

template <typename i_t, typename f_t>
f_t barrier_solver_t<i_t, f_t>::gpu_max_step_to_boundary(iteration_data_t<i_t, f_t>& data,
                                                         const rmm::device_uvector<f_t>& x,
                                                         const rmm::device_uvector<f_t>& dx)
{
  return data.transform_reduce_helper_.transform_reduce(
    thrust::make_zip_iterator(dx.data(), x.data()),
    thrust::minimum<f_t>(),
    [] HD(const thrust::tuple<f_t, f_t> t) {
      const f_t dx = thrust::get<0>(t);
      const f_t x  = thrust::get<1>(t);

      if (dx < f_t(0.0)) return -x / dx;
      return f_t(1.0);
    },
    f_t(1.0),
    x.size(),
    stream_view_);
}

template <typename i_t, typename f_t>
template <typename AllocatorA, typename AllocatorB>
f_t barrier_solver_t<i_t, f_t>::max_step_to_boundary(
  const dense_vector_t<i_t, f_t, AllocatorA>& x,
  const dense_vector_t<i_t, f_t, AllocatorB>& dx) const
{
  i_t index;
  return max_step_to_boundary(x, dx, index);
}

template <typename i_t, typename f_t>
i_t barrier_solver_t<i_t, f_t>::gpu_compute_search_direction(iteration_data_t<i_t, f_t>& data,
                                                             pinned_dense_vector_t<i_t, f_t>& dw,
                                                             pinned_dense_vector_t<i_t, f_t>& dx,
                                                             pinned_dense_vector_t<i_t, f_t>& dy,
                                                             pinned_dense_vector_t<i_t, f_t>& dv,
                                                             pinned_dense_vector_t<i_t, f_t>& dz,
                                                             f_t& max_residual)
{
  raft::common::nvtx::range fun_scope("Barrier: compute_search_direction");

  const bool debug = false;

  {
    raft::common::nvtx::range fun_scope("Barrier: GPU allocation and copies");

    // TMP allocation and copy should happen only once where it's written in a first place
    d_bound_rhs_.resize(data.bound_rhs.size(), stream_view_);
    raft::copy(d_bound_rhs_.data(), data.bound_rhs.data(), data.bound_rhs.size(), stream_view_);
    d_x_.resize(data.x.size(), stream_view_);
    raft::copy(d_x_.data(), data.x.data(), data.x.size(), stream_view_);
    d_z_.resize(data.z.size(), stream_view_);
    raft::copy(d_z_.data(), data.z.data(), data.z.size(), stream_view_);
    d_w_.resize(data.w.size(), stream_view_);
    raft::copy(d_w_.data(), data.w.data(), data.w.size(), stream_view_);
    d_v_.resize(data.v.size(), stream_view_);
    raft::copy(d_v_.data(), data.v.data(), data.v.size(), stream_view_);
    d_upper_bounds_.resize(data.upper_bounds.size(), stream_view_);
    raft::copy(
      d_upper_bounds_.data(), data.upper_bounds.data(), data.upper_bounds.size(), stream_view_);
    d_dy_.resize(dy.size(), stream_view_);
    raft::copy(d_dy_.data(), dy.data(), dy.size(), stream_view_);
    d_dx_.resize(dx.size(), stream_view_);
    raft::copy(d_h_.data(), data.primal_rhs.data(), data.primal_rhs.size(), stream_view_);
    raft::copy(d_dual_rhs_.data(), data.dual_rhs.data(), data.dual_rhs.size(), stream_view_);
    d_dz_.resize(dz.size(), stream_view_);
    d_dv_.resize(dv.size(), stream_view_);
    d_dw_.resize(data.bound_rhs.size(), stream_view_);
    raft::copy(d_dw_.data(), data.bound_rhs.data(), data.bound_rhs.size(), stream_view_);
    d_dw_residual_.resize(data.n_upper_bounds, stream_view_);
    d_wv_residual_.resize(d_complementarity_wv_rhs_.size(), stream_view_);
    d_xz_residual_.resize(d_complementarity_xz_rhs_.size(), stream_view_);
    d_primal_residual_.resize(lp.rhs.size(), stream_view_);
    raft::copy(d_primal_residual_.data(), lp.rhs.data(), lp.rhs.size(), stream_view_);
    d_bound_residual_.resize(data.bound_residual.size(), stream_view_);
    d_upper_.resize(lp.upper.size(), stream_view_);
    raft::copy(d_upper_.data(), lp.upper.data(), lp.upper.size(), stream_view_);
  }

  // Solves the linear system
  //
  //  dw dx dy dv dz
  // [ 0 A  0   0  0 ] [ dw ] = [ rp  ]
  // [ I E' 0   0  0 ] [ dx ]   [ rw  ]
  // [ 0 0  A' -E  I ] [ dy ]   [ rd  ]
  // [ 0 Z  0   0  X ] [ dv ]   [ rxz ]
  // [ V 0  0   W  0 ] [ dz ]   [ rwv ]

  max_residual = 0.0;
  {
    raft::common::nvtx::range fun_scope("Barrier: GPU diag, inv diag and sqrt inv diag formation");

    // diag = z ./ x
    cub::DeviceTransform::Transform(cuda::std::make_tuple(d_z_.data(), d_x_.data()),
                                    d_diag_.data(),
                                    d_diag_.size(),
                                    cuda::std::divides<>{},
                                    stream_view_);

    // Returns v / w and diag so that we can reuse it in the output iterator to do diag = diag + v
    // /
    // w
    const auto diag_and_v_divide_w_iterator = thrust::make_transform_iterator(
      thrust::make_zip_iterator(d_v_.data(), d_w_.data(), d_diag_.data()),
      [] HD(const thrust::tuple<double, double, double> t) -> thrust::tuple<f_t, f_t> {
        return {thrust::get<0>(t) / thrust::get<1>(t), thrust::get<2>(t)};
      });

    // diag = z ./ x + E * (v ./ w) * E'
    thrust::scatter(rmm::exec_policy(stream_view_),
                    diag_and_v_divide_w_iterator,
                    diag_and_v_divide_w_iterator + d_upper_bounds_.size(),
                    d_upper_bounds_.data(),
                    thrust::make_transform_output_iterator(
                      d_diag_.data(), [] HD(const thrust::tuple<double, double> t) {
                        return thrust::get<0>(t) + thrust::get<1>(t);
                      }));

    // inv_diag = 1.0 ./ diag
    cub::DeviceTransform::Transform(
      d_diag_.data(),
      data.d_inv_diag.data(),
      d_diag_.size(),
      [] HD(f_t diag) { return f_t(1) / diag; },
      stream_view_);

    raft::copy(data.diag.data(), d_diag_.data(), d_diag_.size(), stream_view_);
    raft::copy(data.inv_diag.data(), data.d_inv_diag.data(), data.d_inv_diag.size(), stream_view_);
  }

  // Form A*D*A'
  if (!data.has_factorization) {
    raft::common::nvtx::range fun_scope("Barrier: ADAT");

    i_t status;
    // compute ADAT = A Dinv * A^T
    data.form_adat();
    // factorize
    if (use_gpu) {
      status = data.chol->factorize(data.device_ADAT);
    } else {
      status = data.chol->factorize(data.ADAT);
    }
    if (status < 0) {
      settings.log.printf("Factorization failed.\n");
      return -1;
    }
    data.has_factorization = true;
    data.num_factorizations++;

    data.has_solve_info = false;
  }

  // Compute h = primal_rhs + A*inv_diag*(dual_rhs - complementarity_xz_rhs ./ x +
  // E*((complementarity_wv_rhs - v .* bound_rhs) ./ w) )
  // TMP shouldn't be allocated when in GPU mode

  {
    raft::common::nvtx::range fun_scope("Barrier: GPU compute H");
    // tmp3 <- E * ((complementarity_wv_rhs .- v .* bound_rhs) ./ w)
    RAFT_CUDA_TRY(cudaMemsetAsync(d_tmp3_.data(), 0, sizeof(f_t) * d_tmp3_.size(), stream_view_));
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        d_bound_rhs_.data(), d_v_.data(), d_complementarity_wv_rhs_.data(), d_w_.data()),
      thrust::make_permutation_iterator(d_tmp3_.data(), d_upper_bounds_.data()),
      data.n_upper_bounds,
      [] HD(f_t bound_rhs, f_t v, f_t complementarity_wv_rhs, f_t w) {
        return (complementarity_wv_rhs - v * bound_rhs) / w;
      },
      stream_view_);

    // tmp3 <- tmp3 .+ -(complementarity_xz_rhs ./ x) .+ dual_rhs
    // tmp4 <- inv_diag .* tmp3
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(data.d_inv_diag.data(),
                            d_tmp3_.data(),
                            d_complementarity_xz_rhs_.data(),
                            d_x_.data(),
                            d_dual_rhs_.data()),
      thrust::make_zip_iterator(d_tmp3_.data(), d_tmp4_.data()),
      lp.num_cols,
      [] HD(f_t inv_diag, f_t tmp3, f_t complementarity_xz_rhs, f_t x, f_t dual_rhs)
        -> thrust::tuple<f_t, f_t> {
        const f_t tmp = tmp3 + -(complementarity_xz_rhs / x) + dual_rhs;
        return {tmp, inv_diag * tmp};
      },
      stream_view_);

    raft::copy(d_r1_.data(), d_tmp3_.data(), d_tmp3_.size(), stream_view_);
    raft::copy(d_r1_prime_.data(), d_tmp3_.data(), d_tmp3_.size(), stream_view_);

    // h <- A @ tmp4 .+ primal_rhs
    data.cusparse_view_.spmv(1, data.cusparse_tmp4_, 1, data.cusparse_h_);
  }

  {
    raft::common::nvtx::range fun_scope("Barrier: Solve A D^{-1} A^T dy = h");

    // Solve A D^{-1} A^T dy = h
    i_t solve_status = data.gpu_solve_adat(d_h_, d_dy_);
    // TODO Chris, we need to write to cpu because dx is used outside
    // Can't we also GPUify what's usinng this dx?
    raft::copy(dy.data(), d_dy_.data(), dy.size(), stream_view_);
    if (solve_status < 0) {
      settings.log.printf("Linear solve failed\n");
      return -1;
    }
  }  // Close NVTX range

  // y_residual <- ADAT*dy - h
  {
    raft::common::nvtx::range fun_scope("Barrier: GPU y_residual");

    raft::copy(d_y_residual_.data(), d_h_.data(), d_h_.size(), stream_view_);

    // TMP should be done only once
    cusparseDnVecDescr_t cusparse_dy_ = data.cusparse_view_.create_vector(d_dy_);

    data.gpu_adat_multiply(1.0,
                           d_dy_,
                           cusparse_dy_,
                           -1.0,
                           d_y_residual_,
                           data.cusparse_y_residual_,
                           d_u_,
                           data.cusparse_u_,
                           data.cusparse_view_,
                           data.d_inv_diag);

    f_t const y_residual_norm = device_vector_norm_inf<i_t, f_t>(d_y_residual_, stream_view_);
    max_residual              = std::max(max_residual, y_residual_norm);
    if (y_residual_norm > 1e-2) {
      settings.log.printf("|| h || = %.2e\n", device_vector_norm_inf<i_t, f_t>(d_h_, stream_view_));
      settings.log.printf("||ADAT*dy - h|| = %.2e\n", y_residual_norm);
    }
    if (y_residual_norm > 10) { return -1; }
  }

  // dx = dinv .* (A'*dy - dual_rhs + complementarity_xz_rhs ./ x  - E *((complementarity_wv_rhs -
  // v
  // .* bound_rhs) ./ w))
  {
    raft::common::nvtx::range fun_scope("Barrier: dx formation GPU");

    // TMP should only be init once
    data.cusparse_dy_ = data.cusparse_view_.create_vector(d_dy_);

    // r1 <- A'*dy - r1
    data.cusparse_view_.transpose_spmv(1.0, data.cusparse_dy_, -1.0, data.cusparse_r1_);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(data.d_inv_diag.data(), d_r1_.data(), d_diag_.data()),
      thrust::make_zip_iterator(d_dx_.data(), d_dx_residual_.data()),
      data.d_inv_diag.size(),
      [] HD(f_t inv_diag, f_t r1, f_t diag) -> thrust::tuple<f_t, f_t> {
        const f_t dx = inv_diag * r1;
        return {dx, dx * diag};
      },
      stream_view_);

    raft::copy(dx.data(), d_dx_.data(), d_dx_.size(), stream_view_);

    data.cusparse_view_.transpose_spmv(-1.0, data.cusparse_dy_, 1.0, data.cusparse_dx_residual_);
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(d_dx_residual_.data(), d_r1_prime_.data()),
      d_dx_residual_.data(),
      d_dx_residual_.size(),
      [] HD(f_t dx_residual, f_t r1_prime) { return dx_residual + r1_prime; },
      stream_view_);
  }

  // Not put on the GPU since debug only
  if (debug) {
    const f_t dx_residual_norm = device_vector_norm_inf<i_t, f_t>(d_dx_residual_, stream_view_);
    max_residual               = std::max(max_residual, dx_residual_norm);
    if (dx_residual_norm > 1e-2) {
      settings.log.printf("|| D * dx - A'*y + r1 || = %.2e\n", dx_residual_norm);
    }
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: dx_residual_2 GPU");

    // norm_inf(D^-1 * (A'*dy - r1) - dx)
    const f_t dx_residual_2_norm = device_custom_vector_norm_inf<i_t, f_t>(
      thrust::make_transform_iterator(
        thrust::make_zip_iterator(data.d_inv_diag.data(), d_r1_.data(), d_dx_.data()),
        [] HD(thrust::tuple<f_t, f_t, f_t> t) -> f_t {
          f_t inv_diag = thrust::get<0>(t);
          f_t r1       = thrust::get<1>(t);
          f_t dx       = thrust::get<2>(t);
          return inv_diag * r1 - dx;
        }),
      d_dx_.size(),
      stream_view_);
    max_residual = std::max(max_residual, dx_residual_2_norm);
    if (dx_residual_2_norm > 1e-2)
      settings.log.printf("|| D^-1 (A'*dy - r1) - dx || = %.2e\n", dx_residual_2_norm);
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: GPU dx_residual_5_6");

    // TMP data should already be on the GPU (not fixed for now since debug only)
    rmm::device_uvector<f_t> d_dx_residual_5(lp.num_cols, stream_view_);
    rmm::device_uvector<f_t> d_dx_residual_6(lp.num_rows, stream_view_);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(data.d_inv_diag.data(), d_r1_.data()),
      d_dx_residual_5.data(),
      d_dx_residual_5.size(),
      [] HD(f_t ind_diag, f_t r1) { return ind_diag * r1; },
      stream_view_);

    // TMP should be done just one in the constructor
    data.cusparse_dx_residual_5_ = data.cusparse_view_.create_vector(d_dx_residual_5);
    data.cusparse_dx_residual_6_ = data.cusparse_view_.create_vector(d_dx_residual_6);
    data.cusparse_dx_            = data.cusparse_view_.create_vector(d_dx_);

    data.cusparse_view_.spmv(1.0, data.cusparse_dx_residual_5_, 0.0, data.cusparse_dx_residual_6_);
    data.cusparse_view_.spmv(-1.0, data.cusparse_dx_, 1.0, data.cusparse_dx_residual_6_);

    const f_t dx_residual_6_norm = device_vector_norm_inf<i_t, f_t>(d_dx_residual_6, stream_view_);
    max_residual                 = std::max(max_residual, dx_residual_6_norm);
    if (dx_residual_6_norm > 1e-2) {
      settings.log.printf("|| A * D^-1 (A'*dy - r1) - A * dx || = %.2e\n", dx_residual_6_norm);
    }
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: GPU dx_residual_3_4");

    // TMP data should already be on the GPU
    rmm::device_uvector<f_t> d_dx_residual_3(lp.num_cols, stream_view_);
    rmm::device_uvector<f_t> d_dx_residual_4(lp.num_rows, stream_view_);

    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(data.d_inv_diag.data(), d_r1_prime_.data()),
      d_dx_residual_3.data(),
      d_dx_residual_3.size(),
      [] HD(f_t ind_diag, f_t r1_prime) { return ind_diag * r1_prime; },
      stream_view_);

    // TMP vector creation should only be done once
    data.cusparse_dx_residual_3_ = data.cusparse_view_.create_vector(d_dx_residual_3);
    data.cusparse_dx_residual_4_ = data.cusparse_view_.create_vector(d_dx_residual_4);
    data.cusparse_dx_            = data.cusparse_view_.create_vector(d_dx_);

    data.cusparse_view_.spmv(1.0, data.cusparse_dx_residual_3_, 0.0, data.cusparse_dx_residual_4_);
    data.cusparse_view_.spmv(1.0, data.cusparse_dx_, 1.0, data.cusparse_dx_residual_4_);
  }

#if CHECK_FORM_ADAT
  csc_matrix_t<i_t, f_t> ADinv = lp.A;
  ADinv.scale_columns(data.inv_diag);
  csc_matrix_t<i_t, f_t> ADinvAT(lp.num_rows, lp.num_rows, 1);
  csc_matrix_t<i_t, f_t> Atranspose(1, 1, 0);
  lp.A.transpose(Atranspose);
  multiply(ADinv, Atranspose, ADinvAT);
  matrix_vector_multiply(ADinvAT, 1.0, dy, -1.0, dx_residual_4);
  const f_t dx_residual_4_norm = vector_norm_inf<i_t, f_t>(dx_residual_4, stream_view_);
  max_residual                 = std::max(max_residual, dx_residual_4_norm);
  if (dx_residual_4_norm > 1e-2) {
    settings.log.printf("|| ADAT * dy - A * D^-1 * r1 - A * dx || = %.2e\n", dx_residual_4_norm);
  }

  csc_matrix_t<i_t, f_t> C(lp.num_rows, lp.num_rows, 1);
  add(ADinvAT, data.ADAT, 1.0, -1.0, C);
  const f_t matrix_residual = C.norm1();
  max_residual              = std::max(max_residual, matrix_residual);
  if (matrix_residual > 1e-2) {
    settings.log.printf("|| AD^{-1/2} D^{-1} A^T + E - A D^{-1} A^T|| = %.2e\n", matrix_residual);
  }
#endif

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: GPU dx_residual_7");

    // TMP data should already be on the GPU
    rmm::device_uvector<f_t> d_dx_residual_7(d_h_, stream_view_);
    cusparseDnVecDescr_t cusparse_dy_ = data.cusparse_view_.create_vector(d_dy_);
    cusparseDnVecDescr_t cusparse_dx_residual_7 =
      data.cusparse_view_.create_vector(d_dx_residual_7);

    // matrix_vector_multiply(data.ADAT, 1.0, dy, -1.0, dx_residual_7);
    data.gpu_adat_multiply(1.0,
                           d_dy_,
                           cusparse_dy_,
                           -1.0,
                           d_dx_residual_7,
                           cusparse_dx_residual_7,
                           d_u_,
                           data.cusparse_u_,
                           data.cusparse_view_,
                           data.d_inv_diag);

    const f_t dx_residual_7_norm = device_vector_norm_inf<i_t, f_t>(d_dx_residual_7, stream_view_);
    max_residual                 = std::max(max_residual, dx_residual_7_norm);
    if (dx_residual_7_norm > 1e-2) {
      settings.log.printf("|| A D^{-1} A^T * dy - h || = %.2e\n", dx_residual_7_norm);
    }
  }

  // Only debug so not ported to the GPU
  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: x_residual");

    // x_residual <- A * dx - primal_rhs
    // TMP data should only be on the GPU
    pinned_dense_vector_t<i_t, f_t> x_residual = data.primal_rhs;
    if (use_gpu) {
      data.cusparse_view_.spmv(1.0, dx, -1.0, x_residual);
    } else {
      matrix_vector_multiply(lp.A, 1.0, dx, -1.0, x_residual);
    }
    const f_t x_residual_norm = vector_norm_inf<i_t, f_t>(x_residual, stream_view_);
    max_residual              = std::max(max_residual, x_residual_norm);
    if (x_residual_norm > 1e-2) {
      settings.log.printf("|| A * dx - rp || = %.2e\n", x_residual_norm);
    }
  }

  {
    raft::common::nvtx::range fun_scope("Barrier: dz formation GPU");

    // dz = (complementarity_xz_rhs - z.* dx) ./ x;
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        d_complementarity_xz_rhs_.data(), d_z_.data(), d_dx_.data(), d_x_.data()),
      d_dz_.data(),
      d_dz_.size(),
      [] HD(f_t complementarity_xz_rhs, f_t z, f_t dx, f_t x) {
        return (complementarity_xz_rhs - z * dx) / x;
      },
      stream_view_);

    raft::copy(dz.data(), d_dz_.data(), d_dz_.size(), stream_view_);
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: xz_residual GPU");

    // xz_residual <- z .* dx + x .* dz - complementarity_xz_rhs
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        d_complementarity_xz_rhs_.data(), d_z_.data(), d_dz_.data(), d_dx_.data(), d_x_.data()),
      d_xz_residual_.data(),
      d_xz_residual_.size(),
      [] HD(f_t complementarity_xz_rhs, f_t z, f_t dz, f_t dx, f_t x) {
        return z * dx + x * dz - complementarity_xz_rhs;
      },
      stream_view_);
    const f_t xz_residual_norm = device_vector_norm_inf<i_t, f_t>(d_xz_residual_, stream_view_);
    max_residual               = std::max(max_residual, xz_residual_norm);
    if (xz_residual_norm > 1e-2)
      settings.log.printf("|| Z dx + X dz - rxz || = %.2e\n", xz_residual_norm);
  }

  {
    raft::common::nvtx::range fun_scope("Barrier: dv formation GPU");
    // dv <- (v .* E' * dx + complementarity_wv_rhs - v .* bound_rhs) ./ w
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(d_v_.data(),
                            thrust::make_permutation_iterator(d_dx_.data(), d_upper_bounds_.data()),
                            d_bound_rhs_.data(),
                            d_complementarity_wv_rhs_.data(),
                            d_w_.data()),
      d_dv_.data(),
      d_dv_.size(),
      [] HD(f_t v, f_t gathered_dx, f_t bound_rhs, f_t complementarity_wv_rhs, f_t w) {
        return (v * gathered_dx - bound_rhs * v + complementarity_wv_rhs) / w;
      },
      stream_view_);

    raft::copy(dv.data(), d_dv_.data(), d_dv_.size(), stream_view_);
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: dv_residual GPU");

    // TMP data should already be on the GPU (not fixed for now since debug only)
    rmm::device_uvector<f_t> d_dv_residual(data.n_upper_bounds, stream_view_);
    // dv_residual <- -v .* E' * dx + w .* dv - complementarity_wv_rhs + v .* bound_rhs
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(d_v_.data(),
                            thrust::make_permutation_iterator(d_dx_.data(), d_upper_bounds_.data()),
                            d_dv_.data(),
                            d_bound_rhs_.data(),
                            d_complementarity_wv_rhs_.data(),
                            d_w_.data()),
      d_dv_residual.data(),
      d_dv_residual.size(),
      [] HD(f_t v, f_t gathered_dx, f_t dv, f_t bound_rhs, f_t complementarity_wv_rhs, f_t w) {
        return -v * gathered_dx + w * dv - complementarity_wv_rhs + v * bound_rhs;
      },
      stream_view_);

    const f_t dv_residual_norm = device_vector_norm_inf<i_t, f_t>(d_dv_residual, stream_view_);
    max_residual               = std::max(max_residual, dv_residual_norm);
    if (dv_residual_norm > 1e-2) {
      settings.log.printf(
        "|| -v .* E' * dx + w .* dv - complementarity_wv_rhs - v .* bound_rhs || = %.2e\n",
        dv_residual_norm);
    }
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: dual_residual GPU");

    // dual_residual <- A' * dy - E * dv  + dz -  dual_rhs
    thrust::fill(
      rmm::exec_policy(stream_view_), d_dual_residual_.begin(), d_dual_residual_.end(), f_t(0.0));

    // dual_residual <- E * dv
    thrust::scatter(rmm::exec_policy(stream_view_),
                    d_dv_.begin(),
                    d_dv_.end(),
                    d_upper_bounds_.data(),
                    d_dual_residual_.begin());

    // dual_residual <- A' * dy - E * dv
    data.cusparse_view_.transpose_spmv(1.0, data.cusparse_dy_, -1.0, data.cusparse_dual_residual_);

    // dual_residual <- A' * dy - E * dv + dz - dual_rhs
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(d_dual_residual_.data(), d_dz_.data(), d_dual_rhs_.data()),
      d_dual_residual_.data(),
      d_dual_residual_.size(),
      [] HD(f_t dual_residual, f_t dz, f_t dual_rhs) { return dual_residual + dz - dual_rhs; },
      stream_view_);

    const f_t dual_residual_norm = device_vector_norm_inf<i_t, f_t>(d_dual_residual_, stream_view_);
    max_residual                 = std::max(max_residual, dual_residual_norm);
    if (dual_residual_norm > 1e-2) {
      settings.log.printf("|| A' * dy - E * dv  + dz -  dual_rhs || = %.2e\n", dual_residual_norm);
    }
  }

  {
    raft::common::nvtx::range fun_scope("Barrier: dw formation GPU");

    // dw = bound_rhs - E'*dx
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        d_dw_.data(), thrust::make_permutation_iterator(d_dx_.data(), d_upper_bounds_.data())),
      d_dw_.data(),
      d_dw_.size(),
      [] HD(f_t dw, f_t gathered_dx) { return dw - gathered_dx; },
      stream_view_);

    raft::copy(dw.data(), d_dw_.data(), d_dw_.size(), stream_view_);

    if (debug) {
      // dw_residual <- dw + E'*dx - bound_rhs

      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(
          d_dw_.data(),
          thrust::make_permutation_iterator(d_dx_.data(), d_upper_bounds_.data()),
          d_bound_rhs_.data()),
        d_dw_residual_.data(),
        d_dw_residual_.size(),
        [] HD(f_t dw, f_t gathered_dx, f_t bound_rhs) { return dw + gathered_dx - bound_rhs; },
        stream_view_);

      const f_t dw_residual_norm = device_vector_norm_inf<i_t, f_t>(d_dw_residual_, stream_view_);
      max_residual               = std::max(max_residual, dw_residual_norm);
      if (dw_residual_norm > 1e-2) {
        settings.log.printf("|| dw + E'*dx - bound_rhs || = %.2e\n", dw_residual_norm);
      }
    }
  }

  if (debug) {
    raft::common::nvtx::range fun_scope("Barrier: wv_residual GPU");

    // wv_residual <- v .* dw + w .* dv - complementarity_wv_rhs
    cub::DeviceTransform::Transform(
      cuda::std::make_tuple(
        d_complementarity_wv_rhs_.data(), d_w_.data(), d_v_.data(), d_dw_.data(), d_dv_.data()),
      d_wv_residual_.data(),
      d_wv_residual_.size(),
      [] HD(f_t complementarity_wv_rhs, f_t w, f_t v, f_t dw, f_t dv) {
        return v * dw + w * dv - complementarity_wv_rhs;
      },
      stream_view_);

    const f_t wv_residual_norm = device_vector_norm_inf<i_t, f_t>(d_wv_residual_, stream_view_);
    max_residual               = std::max(max_residual, wv_residual_norm);
    if (wv_residual_norm > 1e-2) {
      settings.log.printf("|| V dw + W dv - rwv || = %.2e\n", wv_residual_norm);
    }
  }

  return 0;
}

template <typename i_t, typename f_t>
lp_status_t barrier_solver_t<i_t, f_t>::check_for_suboptimal_solution(
  const barrier_solver_settings_t<i_t, f_t>& options,
  iteration_data_t<i_t, f_t>& data,
  f_t start_time,
  i_t iter,
  f_t norm_b,
  f_t norm_c,
  f_t& primal_objective,
  f_t& relative_primal_residual,
  f_t& relative_dual_residual,
  f_t& relative_complementarity_residual,
  lp_solution_t<i_t, f_t>& solution)
{
  f_t primal_residual_norm;
  f_t dual_residual_norm;
  f_t complementarity_residual_norm;
  if (use_gpu) {
    auto d_w_save = cuopt::device_copy(data.w_save, stream_view_);
    auto d_x_save = cuopt::device_copy(data.x_save, stream_view_);
    auto d_y_save = cuopt::device_copy(data.y_save, stream_view_);
    auto d_v_save = cuopt::device_copy(data.v_save, stream_view_);
    auto d_z_save = cuopt::device_copy(data.z_save, stream_view_);

    gpu_compute_residual_norms(d_w_save,
                               d_x_save,
                               d_y_save,
                               d_v_save,
                               d_z_save,
                               data,
                               primal_residual_norm,
                               dual_residual_norm,
                               complementarity_residual_norm);
  } else {
    compute_residual_norms(data.w_save,
                           data.x_save,
                           data.y_save,
                           data.v_save,
                           data.z_save,
                           data,
                           primal_residual_norm,
                           dual_residual_norm,
                           complementarity_residual_norm);
  }
  if (relative_primal_residual < 100 * options.feasibility_tol &&
      relative_dual_residual < 100 * options.optimality_tol &&
      relative_complementarity_residual < 100 * options.complementarity_tol) {
    data.to_solution(lp,
                     iter,
                     primal_objective,
                     primal_objective + lp.obj_constant,
                     vector_norm2<i_t, f_t>(data.primal_residual),
                     vector_norm2<i_t, f_t>(data.dual_residual),
                     data.cusparse_view_,
                     solution);
    settings.log.printf(
      "Suboptimal solution found in %d iterations and %.2f seconds\n", iter, toc(start_time));
    settings.log.printf("Objective %+.8e\n", primal_objective + lp.obj_constant);
    settings.log.printf("Primal infeasibility (abs/rel): %8.2e/%8.2e\n",
                        primal_residual_norm,
                        relative_primal_residual);
    settings.log.printf(
      "Dual infeasibility   (abs/rel): %8.2e/%8.2e\n", dual_residual_norm, relative_dual_residual);
    settings.log.printf("Complementarity gap  (abs/rel): %8.2e/%8.2e\n",
                        complementarity_residual_norm,
                        relative_complementarity_residual);
    return lp_status_t::OPTIMAL;  // TODO: Barrier should probably have a separate suboptimal
                                  // status
  }

  primal_objective         = data.c.inner_product(data.x_save);
  relative_primal_residual = primal_residual_norm / (1.0 + norm_b);
  relative_dual_residual   = dual_residual_norm / (1.0 + norm_c);
  relative_complementarity_residual =
    complementarity_residual_norm / (1.0 + std::abs(primal_objective));

  if (relative_primal_residual < 100 * options.feasibility_tol &&
      relative_dual_residual < 100 * options.optimality_tol &&
      relative_complementarity_residual < 100 * options.complementarity_tol) {
    settings.log.printf("Restoring previous solution: primal %.2e dual %.2e complementarity %.2e\n",
                        relative_primal_residual,
                        relative_dual_residual,
                        relative_complementarity_residual);
    data.to_solution(lp,
                     iter,
                     primal_objective,
                     primal_objective + lp.obj_constant,
                     vector_norm2<i_t, f_t>(data.primal_residual),
                     vector_norm2<i_t, f_t>(data.dual_residual),
                     data.cusparse_view_,
                     solution);
    settings.log.printf(
      "Suboptimal solution found in %d iterations and %.2f seconds\n", iter, toc(start_time));
    settings.log.printf("Objective %+.8e\n", primal_objective + lp.obj_constant);
    settings.log.printf("Primal infeasibility (abs/rel): %8.2e/%8.2e\n",
                        primal_residual_norm,
                        relative_primal_residual);
    settings.log.printf(
      "Dual infeasibility   (abs/rel): %8.2e/%8.2e\n", dual_residual_norm, relative_dual_residual);
    settings.log.printf("Complementarity gap  (abs/rel): %8.2e/%8.2e\n",
                        complementarity_residual_norm,
                        relative_complementarity_residual);
    return lp_status_t::OPTIMAL;  // TODO: Barrier should probably have a separate suboptimal
                                  // status
  } else {
    settings.log.printf("Primal residual %.2e dual residual %.2e complementarity residual %.2e\n",
                        relative_primal_residual,
                        relative_dual_residual,
                        relative_complementarity_residual);
  }
  settings.log.printf("Search direction computation failed\n");
  return lp_status_t::NUMERICAL_ISSUES;
}

template <typename i_t, typename f_t>
lp_status_t barrier_solver_t<i_t, f_t>::solve(const barrier_solver_settings_t<i_t, f_t>& options,
                                              lp_solution_t<i_t, f_t>& solution)
{
  raft::common::nvtx::range fun_scope("Barrier: solve");

  float64_t start_time = tic();

  i_t n = lp.num_cols;
  i_t m = lp.num_rows;

  solution.resize(m, n);
  settings.log.printf(
    "Barrier solver: %d constraints, %d variables, %ld nonzeros\n", m, n, lp.A.col_start[n]);

  // Compute the number of free variables
  i_t num_free_variables = presolve_info.free_variable_pairs.size() / 2;
  settings.log.printf("%d free variables\n", num_free_variables);

  // Compute the number of upper bounds
  i_t num_upper_bounds = 0;
  for (i_t j = 0; j < n; j++) {
    if (lp.upper[j] < inf) { num_upper_bounds++; }
  }

  iteration_data_t<i_t, f_t> data(lp, num_upper_bounds, settings);
  data.cusparse_dual_residual_ = data.cusparse_view_.create_vector(d_dual_residual_);
  data.cusparse_r1_            = data.cusparse_view_.create_vector(d_r1_);
  data.cusparse_tmp4_          = data.cusparse_view_.create_vector(d_tmp4_);
  data.cusparse_h_             = data.cusparse_view_.create_vector(d_h_);
  data.cusparse_dx_residual_   = data.cusparse_view_.create_vector(d_dx_residual_);
  data.cusparse_u_             = data.cusparse_view_.create_vector(d_u_);
  data.cusparse_y_residual_    = data.cusparse_view_.create_vector(d_y_residual_);
  restrict_u_.resize(num_upper_bounds);

  if (toc(start_time) > settings.time_limit) {
    settings.log.printf("Barrier time limit exceeded\n");
    return lp_status_t::TIME_LIMIT;
  }
  settings.log.printf("%d finite upper bounds\n", num_upper_bounds);

  initial_point(data);
  if (toc(start_time) > settings.time_limit) {
    settings.log.printf("Barrier time limit exceeded\n");
    return lp_status_t::TIME_LIMIT;
  }
  if (settings.concurrent_halt != nullptr &&
      settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
    settings.log.printf("Barrier solver halted\n");
    return lp_status_t::CONCURRENT_LIMIT;
  }
  compute_residuals<PinnedHostAllocator<f_t>>(data.w, data.x, data.y, data.v, data.z, data);

  f_t primal_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.primal_residual, stream_view_),
                                      vector_norm_inf<i_t, f_t>(data.bound_residual, stream_view_));
  f_t dual_residual_norm   = vector_norm_inf<i_t, f_t>(data.dual_residual, stream_view_);
  f_t complementarity_residual_norm =
    std::max(vector_norm_inf<i_t, f_t>(data.complementarity_xz_residual, stream_view_),
             vector_norm_inf<i_t, f_t>(data.complementarity_wv_residual, stream_view_));
  f_t mu = (data.complementarity_xz_residual.sum() + data.complementarity_wv_residual.sum()) /
           (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));

  f_t norm_b = vector_norm_inf<i_t, f_t>(data.b, stream_view_);
  f_t norm_c = vector_norm_inf<i_t, f_t>(data.c, stream_view_);

  f_t primal_objective = data.c.inner_product(data.x);

  f_t relative_primal_residual = primal_residual_norm / (1.0 + norm_b);
  f_t relative_dual_residual   = dual_residual_norm / (1.0 + norm_c);
  f_t relative_complementarity_residual =
    complementarity_residual_norm / (1.0 + std::abs(primal_objective));

  dense_vector_t<i_t, f_t> upper(lp.upper);
  data.gather_upper_bounds(upper, restrict_u_);
  f_t dual_objective = data.b.inner_product(data.y) - restrict_u_.inner_product(data.v);

  i_t iter = 0;
  settings.log.printf(
    "         Objective                                  Residual             Step-Length     "
    "Time\n");
  settings.log.printf(
    "Iter    Primal               Dual            Primal   Dual    Compl.     Primal Dual     "
    "Elapsed\n");
  float64_t elapsed_time = toc(start_time);
  settings.log.printf("%2d   %+.12e %+.12e %.2e %.2e %.2e                   %.1f\n",
                      iter,
                      primal_objective,
                      dual_objective,
                      primal_residual_norm,
                      dual_residual_norm,
                      complementarity_residual_norm,
                      elapsed_time);

  bool converged = primal_residual_norm < options.feasibility_tol &&
                   dual_residual_norm < options.optimality_tol &&
                   complementarity_residual_norm < options.complementarity_tol;

  d_complementarity_xz_residual_.resize(data.complementarity_xz_residual.size(), stream_view_);
  d_complementarity_wv_residual_.resize(data.complementarity_wv_residual.size(), stream_view_);
  d_complementarity_xz_rhs_.resize(data.complementarity_xz_rhs.size(), stream_view_);
  d_complementarity_wv_rhs_.resize(data.complementarity_wv_rhs.size(), stream_view_);
  raft::copy(d_complementarity_xz_residual_.data(),
             data.complementarity_xz_residual.data(),
             data.complementarity_xz_residual.size(),
             stream_view_);
  raft::copy(d_complementarity_wv_residual_.data(),
             data.complementarity_wv_residual.data(),
             data.complementarity_wv_residual.size(),
             stream_view_);
  raft::copy(d_complementarity_xz_rhs_.data(),
             data.complementarity_xz_rhs.data(),
             data.complementarity_xz_rhs.size(),
             stream_view_);
  raft::copy(d_complementarity_wv_rhs_.data(),
             data.complementarity_wv_rhs.data(),
             data.complementarity_wv_rhs.size(),
             stream_view_);

  data.w_save = data.w;
  data.x_save = data.x;
  data.y_save = data.y;
  data.v_save = data.v;
  data.z_save = data.z;

  const i_t iteration_limit = std::min(options.iteration_limit, settings.iteration_limit);

  while (iter < iteration_limit) {
    raft::common::nvtx::range fun_scope("Barrier: iteration");

    {
      raft::common::nvtx::range fun_scope("Barrier: before first compute_search_direction");

      if (toc(start_time) > settings.time_limit) {
        settings.log.printf("Barrier time limit exceeded\n");
        return lp_status_t::TIME_LIMIT;
      }
      if (settings.concurrent_halt != nullptr &&
          settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
        settings.log.printf("Barrier solver halted\n");
        return lp_status_t::CONCURRENT_LIMIT;
      }
      // Compute the affine step
      data.primal_rhs = data.primal_residual;
      data.bound_rhs  = data.bound_residual;
      data.dual_rhs   = data.dual_residual;

      if (use_gpu) {
        raft::copy(d_complementarity_xz_rhs_.data(),
                   d_complementarity_xz_residual_.data(),
                   d_complementarity_xz_residual_.size(),
                   stream_view_);
        raft::copy(d_complementarity_wv_rhs_.data(),
                   d_complementarity_wv_residual_.data(),
                   d_complementarity_wv_residual_.size(),
                   stream_view_);

        // x.*z ->  -x .* z
        cub::DeviceTransform::Transform(
          d_complementarity_xz_rhs_.data(),
          d_complementarity_xz_rhs_.data(),
          d_complementarity_xz_rhs_.size(),
          [] HD(f_t xz_rhs) { return -xz_rhs; },
          stream_view_);

        // w.*v -> -w .* v
        cub::DeviceTransform::Transform(
          d_complementarity_wv_rhs_.data(),
          d_complementarity_wv_rhs_.data(),
          d_complementarity_wv_rhs_.size(),
          [] HD(f_t wv_rhs) { return -wv_rhs; },
          stream_view_);
      } else {
        data.complementarity_xz_rhs = data.complementarity_xz_residual;
        data.complementarity_wv_rhs = data.complementarity_wv_residual;

        // x.*z ->  -x .* z
        data.complementarity_xz_rhs.multiply_scalar(-1.0);
        // w.*v -> -w .* v
        data.complementarity_wv_rhs.multiply_scalar(-1.0);
      }
    }  // Close nvtx range

    f_t max_affine_residual = 0.0;

    i_t status = use_gpu ? gpu_compute_search_direction(data,
                                                        data.dw_aff,
                                                        data.dx_aff,
                                                        data.dy_aff,
                                                        data.dv_aff,
                                                        data.dz_aff,
                                                        max_affine_residual)
                         : compute_search_direction(data,
                                                    data.dw_aff,
                                                    data.dx_aff,
                                                    data.dy_aff,
                                                    data.dv_aff,
                                                    data.dz_aff,
                                                    max_affine_residual);
    // Sync to make sure all the async copies to host done inside are finished
    if (use_gpu) RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

    if (status < 0) {
      return check_for_suboptimal_solution(options,
                                           data,
                                           start_time,
                                           iter,
                                           norm_b,
                                           norm_c,
                                           primal_objective,
                                           relative_primal_residual,
                                           relative_dual_residual,
                                           relative_complementarity_residual,
                                           solution);
    }
    if (toc(start_time) > settings.time_limit) {
      settings.log.printf("Barrier time limit exceeded\n");
      return lp_status_t::TIME_LIMIT;
    }
    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      settings.log.printf("Barrier solver halted\n");
      return lp_status_t::CONCURRENT_LIMIT;
    }

    if (!use_gpu) {
      raft::common::nvtx::range fun_scope("Barrier: CPU after first compute_search_direction");

      f_t step_primal_aff = std::min(max_step_to_boundary(data.w, data.dw_aff),
                                     max_step_to_boundary(data.x, data.dx_aff));
      f_t step_dual_aff   = std::min(max_step_to_boundary(data.v, data.dv_aff),
                                   max_step_to_boundary(data.z, data.dz_aff));

      // w_aff = w + step_primal_aff * dw_aff
      // x_aff = x + step_primal_aff * dx_aff
      // v_aff = v + step_dual_aff * dv_aff
      // z_aff = z + step_dual_aff * dz_aff
      dense_vector_t<i_t, f_t> w_aff = data.w;
      dense_vector_t<i_t, f_t> x_aff = data.x;
      dense_vector_t<i_t, f_t> v_aff = data.v;
      dense_vector_t<i_t, f_t> z_aff = data.z;
      w_aff.axpy(step_primal_aff, data.dw_aff, 1.0);
      x_aff.axpy(step_primal_aff, data.dx_aff, 1.0);
      v_aff.axpy(step_dual_aff, data.dv_aff, 1.0);
      z_aff.axpy(step_dual_aff, data.dz_aff, 1.0);

      dense_vector_t<i_t, f_t> complementarity_xz_aff(lp.num_cols);
      dense_vector_t<i_t, f_t> complementarity_wv_aff(num_upper_bounds);
      x_aff.pairwise_product(z_aff, complementarity_xz_aff);
      w_aff.pairwise_product(v_aff, complementarity_wv_aff);

      f_t complementarity_aff_sum = complementarity_xz_aff.sum() + complementarity_wv_aff.sum();
      f_t mu_aff =
        (complementarity_aff_sum) / (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
      f_t sigma  = std::max(0.0, std::min(1.0, std::pow(mu_aff / mu, 3.0)));
      f_t new_mu = sigma * mu_aff;

      // complementarity_xz_rhs = -dx_aff .* dz_aff + sigma * mu
      data.dx_aff.pairwise_product(data.dz_aff, data.complementarity_xz_rhs);
      data.complementarity_xz_rhs.multiply_scalar(-1.0);
      data.complementarity_xz_rhs.add_scalar(new_mu);

      // complementarity_wv_rhs = -dw_aff .* dv_aff + sigma * mu
      data.dw_aff.pairwise_product(data.dv_aff, data.complementarity_wv_rhs);
      data.complementarity_wv_rhs.multiply_scalar(-1.0);
      data.complementarity_wv_rhs.add_scalar(new_mu);
    } else {
      raft::common::nvtx::range fun_scope("Barrier: GPU after first compute_search_direction");

      // TMP no copy and data should always be on the GPU
      d_dw_aff_.resize(data.dw_aff.size(), stream_view_);
      d_dx_aff_.resize(data.dx_aff.size(), stream_view_);
      d_dv_aff_.resize(data.dv_aff.size(), stream_view_);
      d_dz_aff_.resize(data.dz_aff.size(), stream_view_);

      raft::copy(d_dw_aff_.data(), data.dw_aff.data(), data.dw_aff.size(), stream_view_);
      raft::copy(d_dx_aff_.data(), data.dx_aff.data(), data.dx_aff.size(), stream_view_);
      raft::copy(d_dv_aff_.data(), data.dv_aff.data(), data.dv_aff.size(), stream_view_);
      raft::copy(d_dz_aff_.data(), data.dz_aff.data(), data.dz_aff.size(), stream_view_);

      f_t step_primal_aff = std::min(gpu_max_step_to_boundary(data, d_w_, d_dw_aff_),
                                     gpu_max_step_to_boundary(data, d_x_, d_dx_aff_));
      f_t step_dual_aff   = std::min(gpu_max_step_to_boundary(data, d_v_, d_dv_aff_),
                                   gpu_max_step_to_boundary(data, d_z_, d_dz_aff_));

      f_t complementarity_xz_aff_sum = data.transform_reduce_helper_.transform_reduce(
        thrust::make_zip_iterator(d_x_.data(), d_z_.data(), d_dx_aff_.data(), d_dz_aff_.data()),
        cuda::std::plus<f_t>{},
        [step_primal_aff, step_dual_aff] HD(const thrust::tuple<f_t, f_t, f_t, f_t> t) {
          const f_t x      = thrust::get<0>(t);
          const f_t z      = thrust::get<1>(t);
          const f_t dx_aff = thrust::get<2>(t);
          const f_t dz_aff = thrust::get<3>(t);

          const f_t x_aff = x + step_primal_aff * dx_aff;
          const f_t z_aff = z + step_dual_aff * dz_aff;

          const f_t complementarity_xz_aff = x_aff * z_aff;

          return complementarity_xz_aff;
        },
        f_t(0),
        d_x_.size(),
        stream_view_);

      f_t complementarity_wv_aff_sum = data.transform_reduce_helper_.transform_reduce(
        thrust::make_zip_iterator(d_w_.data(), d_v_.data(), d_dw_aff_.data(), d_dv_aff_.data()),
        cuda::std::plus<f_t>{},
        [step_primal_aff, step_dual_aff] HD(const thrust::tuple<f_t, f_t, f_t, f_t> t) {
          const f_t w      = thrust::get<0>(t);
          const f_t v      = thrust::get<1>(t);
          const f_t dw_aff = thrust::get<2>(t);
          const f_t dv_aff = thrust::get<3>(t);

          const f_t w_aff = w + step_primal_aff * dw_aff;
          const f_t v_aff = v + step_dual_aff * dv_aff;

          const f_t complementarity_wv_aff = w_aff * v_aff;

          return complementarity_wv_aff;
        },
        f_t(0),
        d_w_.size(),
        stream_view_);

      f_t complementarity_aff_sum = complementarity_xz_aff_sum + complementarity_wv_aff_sum;

      f_t mu_aff =
        (complementarity_aff_sum) / (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
      f_t sigma  = std::max(0.0, std::min(1.0, std::pow(mu_aff / mu, 3.0)));
      f_t new_mu = sigma * mu_aff;

      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(d_dx_aff_.data(), d_dz_aff_.data()),
        d_complementarity_xz_rhs_.data(),
        d_complementarity_xz_rhs_.size(),
        [new_mu] HD(f_t dx_aff, f_t dz_aff) { return -(dx_aff * dz_aff) + new_mu; },
        stream_view_);

      cub::DeviceTransform::Transform(
        cuda::std::make_tuple(d_dw_aff_.data(), d_dv_aff_.data()),
        d_complementarity_wv_rhs_.data(),
        d_complementarity_wv_rhs_.size(),
        [new_mu] HD(f_t dw_aff, f_t dv_aff) { return -(dw_aff * dv_aff) + new_mu; },
        stream_view_);
    }

    // TMP should be CPU to 0 if CPU and GPU to 0 if GPU
    data.primal_rhs.set_scalar(0.0);
    data.bound_rhs.set_scalar(0.0);
    data.dual_rhs.set_scalar(0.0);

    f_t max_corrector_residual = 0.0;
    status                     = use_gpu ? gpu_compute_search_direction(
                         data, data.dw, data.dx, data.dy, data.dv, data.dz, max_corrector_residual)
                                         : compute_search_direction(
                         data, data.dw, data.dx, data.dy, data.dv, data.dz, max_corrector_residual);
    // Sync to make sure all the async copies to host done inside are finished
    if (use_gpu) RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));

    {
      raft::common::nvtx::range fun_scope("Barrier: after second compute_search_direction");

      if (status < 0) {
        return check_for_suboptimal_solution(options,
                                             data,
                                             start_time,
                                             iter,
                                             norm_b,
                                             norm_c,
                                             primal_objective,
                                             relative_primal_residual,
                                             relative_dual_residual,
                                             relative_complementarity_residual,
                                             solution);
      }
      data.has_factorization = false;
      data.has_solve_info    = false;
      if (toc(start_time) > settings.time_limit) {
        settings.log.printf("Barrier time limit exceeded\n");
        return lp_status_t::TIME_LIMIT;
      }
      if (settings.concurrent_halt != nullptr &&
          settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
        settings.log.printf("Barrier solver halted\n");
        return lp_status_t::CONCURRENT_LIMIT;
      }

      f_t step_primal, step_dual;

      if (use_gpu) {
        raft::common::nvtx::range fun_scope("Barrier: GPU vector operations");
        // TODO Nicolas: Redundant copies
        d_y_.resize(data.y.size(), stream_view_);
        d_dy_aff_.resize(data.dy_aff.size(), stream_view_);
        raft::copy(d_y_.data(), data.y.data(), data.y.size(), stream_view_);
        raft::copy(d_dy_aff_.data(), data.dy_aff.data(), data.dy_aff.size(), stream_view_);

        // dw = dw_aff + dw_cc
        // dx = dx_aff + dx_cc
        // dy = dy_aff + dy_cc
        // dv = dv_aff + dv_cc
        // dz = dz_aff + dz_cc
        // Note: dw_cc - dz_cc are stored in dw - dz

        // Transforms are grouped according to vector sizes.
        assert(d_dw_.size() == d_dv_.size());
        assert(d_dx_.size() == d_dz_.size());
        assert(d_dw_aff_.size() == d_dv_aff_.size());
        assert(d_dx_aff_.size() == d_dz_aff_.size());
        assert(d_dy_aff_.size() == d_dy_.size());

        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(d_dw_aff_.data(), d_dv_aff_.data(), d_dw_.data(), d_dv_.data()),
          thrust::make_zip_iterator(d_dw_.data(), d_dv_.data()),
          d_dw_.size(),
          [] HD(f_t dw_aff, f_t dv_aff, f_t dw, f_t dv) -> thrust::tuple<f_t, f_t> {
            return {dw + dw_aff, dv + dv_aff};
          },
          stream_view_);

        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(d_dx_aff_.data(), d_dz_aff_.data(), d_dx_.data(), d_dz_.data()),
          thrust::make_zip_iterator(d_dx_.data(), d_dz_.data()),
          d_dx_.size(),
          [] HD(f_t dx_aff, f_t dz_aff, f_t dx, f_t dz) -> thrust::tuple<f_t, f_t> {
            return {dx + dx_aff, dz + dz_aff};
          },
          stream_view_);

        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(d_dy_aff_.data(), d_dy_.data()),
          d_dy_.data(),
          d_dy_.size(),
          [] HD(f_t dy_aff, f_t dy) { return dy + dy_aff; },
          stream_view_);

        f_t max_step_primal = std::min(gpu_max_step_to_boundary(data, d_w_, d_dw_),
                                       gpu_max_step_to_boundary(data, d_x_, d_dx_));
        f_t max_step_dual   = std::min(gpu_max_step_to_boundary(data, d_v_, d_dv_),
                                     gpu_max_step_to_boundary(data, d_z_, d_dz_));
        step_primal         = options.step_scale * max_step_primal;
        step_dual           = options.step_scale * max_step_dual;

        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(d_w_.data(), d_v_.data(), d_dw_.data(), d_dv_.data()),
          thrust::make_zip_iterator(d_w_.data(), d_v_.data()),
          d_dw_.size(),
          [step_primal, step_dual] HD(f_t w, f_t v, f_t dw, f_t dv) -> thrust::tuple<f_t, f_t> {
            return {w + step_primal * dw, v + step_dual * dv};
          },
          stream_view_);

        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(d_x_.data(), d_z_.data(), d_dx_.data(), d_dz_.data()),
          thrust::make_zip_iterator(d_x_.data(), d_z_.data()),
          d_dx_.size(),
          [step_primal, step_dual] HD(f_t x, f_t z, f_t dx, f_t dz) -> thrust::tuple<f_t, f_t> {
            return {x + step_primal * dx, z + step_dual * dz};
          },
          stream_view_);

        cub::DeviceTransform::Transform(
          cuda::std::make_tuple(d_y_.data(), d_dy_.data()),
          d_y_.data(),
          d_y_.size(),
          [step_dual] HD(f_t y, f_t dy) { return y + step_dual * dy; },
          stream_view_);

        if (num_free_variables > 0) {
          auto d_free_variable_pairs = device_copy(presolve_info.free_variable_pairs, stream_view_);
          thrust::for_each(rmm::exec_policy(stream_view_),
                           thrust::make_counting_iterator(0),
                           thrust::make_counting_iterator(num_free_variables),
                           [span_free_variable_pairs = cuopt::make_span(d_free_variable_pairs),
                            span_x                   = cuopt::make_span(d_x_),
                            step_scale               = options.step_scale] __device__(i_t i) {
                             // Not coalesced
                             i_t k       = 2 * i;
                             i_t u       = span_free_variable_pairs[k];
                             i_t v       = span_free_variable_pairs[k + 1];
                             f_t u_val   = span_x[u];
                             f_t v_val   = span_x[v];
                             f_t min_val = cuda::std::min(u_val, v_val);
                             f_t eta     = step_scale * min_val;
                             span_x[u] -= eta;
                             span_x[v] -= eta;
                           });
        }

        raft::copy(data.w.data(), d_w_.data(), d_w_.size(), stream_view_);
        raft::copy(data.x.data(), d_x_.data(), d_x_.size(), stream_view_);
        raft::copy(data.y.data(), d_y_.data(), d_y_.size(), stream_view_);
        raft::copy(data.v.data(), d_v_.data(), d_v_.size(), stream_view_);
        raft::copy(data.z.data(), d_z_.data(), d_z_.size(), stream_view_);
        // Sync to make sure all host variable are done copying
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view_));
      } else {
        raft::common::nvtx::range fun_scope("Barrier: CPU vector operations");
        // dw = dw_aff + dw_cc
        // dx = dx_aff + dx_cc
        // dy = dy_aff + dy_cc
        // dv = dv_aff + dv_cc
        // dz = dz_aff + dz_cc
        // Note: dw_cc - dz_cc are stored in dw - dz
        data.dw.axpy(1.0, data.dw_aff, 1.0);
        data.dx.axpy(1.0, data.dx_aff, 1.0);
        data.dy.axpy(1.0, data.dy_aff, 1.0);
        data.dv.axpy(1.0, data.dv_aff, 1.0);
        data.dz.axpy(1.0, data.dz_aff, 1.0);

        f_t max_step_primal =
          std::min(max_step_to_boundary(data.w, data.dw), max_step_to_boundary(data.x, data.dx));
        f_t max_step_dual =
          std::min(max_step_to_boundary(data.v, data.dv), max_step_to_boundary(data.z, data.dz));

        step_primal = options.step_scale * max_step_primal;
        step_dual   = options.step_scale * max_step_dual;

        data.w.axpy(step_primal, data.dw, 1.0);
        data.x.axpy(step_primal, data.dx, 1.0);
        data.y.axpy(step_dual, data.dy, 1.0);
        data.v.axpy(step_dual, data.dv, 1.0);
        data.z.axpy(step_dual, data.dz, 1.0);

        // Handle free variables
        if (num_free_variables > 0) {
          for (i_t k = 0; k < 2 * num_free_variables; k += 2) {
            i_t u       = presolve_info.free_variable_pairs[k];
            i_t v       = presolve_info.free_variable_pairs[k + 1];
            f_t u_val   = data.x[u];
            f_t v_val   = data.x[v];
            f_t min_val = std::min(u_val, v_val);
            f_t eta     = options.step_scale * min_val;
            data.x[u] -= eta;
            data.x[v] -= eta;
          }
        }
      }

      if (use_gpu) {
        gpu_compute_residual_norms(d_w_,
                                   d_x_,
                                   d_y_,
                                   d_v_,
                                   d_z_,
                                   data,
                                   primal_residual_norm,
                                   dual_residual_norm,
                                   complementarity_residual_norm);
      } else {
        compute_residual_norms(data.w,
                               data.x,
                               data.y,
                               data.v,
                               data.z,
                               data,
                               primal_residual_norm,
                               dual_residual_norm,
                               complementarity_residual_norm);
      }

      if (use_gpu) {
        mu = (data.sum_reduce_helper_.sum(d_complementarity_xz_residual_.begin(),
                                          d_complementarity_xz_residual_.size(),
                                          stream_view_) +
              data.sum_reduce_helper_.sum(d_complementarity_wv_residual_.begin(),
                                          d_complementarity_wv_residual_.size(),
                                          stream_view_)) /
             (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
      } else {
        mu = (data.complementarity_xz_residual.sum() + data.complementarity_wv_residual.sum()) /
             (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
      }

      if (use_gpu) {
        raft::common::nvtx::range fun_scope("Barrier: GPU dots");

        raft::copy(d_c_.data(), data.c.data(), data.c.size(), stream_view_);
        auto d_b          = device_copy(data.b, stream_view_);
        auto d_restrict_u = device_copy(restrict_u_, stream_view_);
        rmm::device_scalar<f_t> d_cx(stream_view_);
        rmm::device_scalar<f_t> d_by(stream_view_);
        rmm::device_scalar<f_t> d_uv(stream_view_);

        RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(lp.handle_ptr->get_cublas_handle(),
                                                        d_c_.size(),
                                                        d_c_.data(),
                                                        1,
                                                        d_x_.data(),
                                                        1,
                                                        d_cx.data(),
                                                        stream_view_));
        RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(lp.handle_ptr->get_cublas_handle(),
                                                        d_b.size(),
                                                        d_b.data(),
                                                        1,
                                                        d_y_.data(),
                                                        1,
                                                        d_by.data(),
                                                        stream_view_));
        RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(lp.handle_ptr->get_cublas_handle(),
                                                        d_restrict_u.size(),
                                                        d_restrict_u.data(),
                                                        1,
                                                        d_v_.data(),
                                                        1,
                                                        d_uv.data(),
                                                        stream_view_));
        primal_objective = d_cx.value(stream_view_);
        dual_objective   = d_by.value(stream_view_) - d_uv.value(stream_view_);
      } else {
        primal_objective = data.c.inner_product(data.x);
        dual_objective   = data.b.inner_product(data.y) - restrict_u_.inner_product(data.v);
      }

      relative_primal_residual = primal_residual_norm / (1.0 + norm_b);
      relative_dual_residual   = dual_residual_norm / (1.0 + norm_c);
      relative_complementarity_residual =
        complementarity_residual_norm / (1.0 + std::abs(primal_objective));

      if (relative_primal_residual < 100 * options.feasibility_tol &&
          relative_dual_residual < 100 * options.optimality_tol &&
          relative_complementarity_residual < 100 * options.complementarity_tol) {
        data.w_save = data.w;
        data.x_save = data.x;
        data.y_save = data.y;
        data.v_save = data.v;
        data.z_save = data.z;
      }

      iter++;
      elapsed_time = toc(start_time);

      if (primal_objective != primal_objective || dual_objective != dual_objective) {
        settings.log.printf("Numerical error in objective\n");
        return lp_status_t::NUMERICAL_ISSUES;
      }

      settings.log.printf("%2d   %+.12e %+.12e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.2e %.1f\n",
                          iter,
                          primal_objective + lp.obj_constant,
                          dual_objective + lp.obj_constant,
                          relative_primal_residual,
                          relative_dual_residual,
                          relative_complementarity_residual,
                          step_primal,
                          step_dual,
                          std::min(data.complementarity_xz_residual.minimum(),
                                   data.complementarity_wv_residual.minimum()),
                          mu,
                          std::max(max_affine_residual, max_corrector_residual),
                          elapsed_time);

    }  // Close nvtx range

    bool primal_feasible = relative_primal_residual < options.feasibility_tol;
    bool dual_feasible   = relative_dual_residual < options.optimality_tol;
    bool small_gap       = relative_complementarity_residual < options.complementarity_tol;

    converged = primal_feasible && dual_feasible && small_gap;

    if (converged) {
      settings.log.printf(
        "Optimal solution found in %d iterations and %.2fs\n", iter, toc(start_time));
      settings.log.printf("Objective %+.8e\n", primal_objective + lp.obj_constant);
      settings.log.printf("Primal infeasibility (abs/rel): %8.2e/%8.2e\n",
                          primal_residual_norm,
                          relative_primal_residual);
      settings.log.printf("Dual infeasibility   (abs/rel): %8.2e/%8.2e\n",
                          dual_residual_norm,
                          relative_dual_residual);
      settings.log.printf("Complementarity gap  (abs/rel): %8.2e/%8.2e\n",
                          complementarity_residual_norm,
                          relative_complementarity_residual);
      data.to_solution(lp,
                       iter,
                       primal_objective,
                       primal_objective + lp.obj_constant,
                       primal_residual_norm,
                       dual_residual_norm,
                       data.cusparse_view_,
                       solution);
      return lp_status_t::OPTIMAL;
    }
  }
  data.to_solution(lp,
                   iter,
                   primal_objective,
                   primal_objective + lp.obj_constant,
                   vector_norm2<i_t, f_t>(data.primal_residual),
                   vector_norm2<i_t, f_t>(data.dual_residual),
                   data.cusparse_view_,
                   solution);
  return lp_status_t::ITERATION_LIMIT;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class barrier_solver_t<int, double>;
template class sparse_cholesky_base_t<int, double>;
template class sparse_cholesky_cudss_t<int, double>;
template class iteration_data_t<int, double>;
template class barrier_solver_settings_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::dual_simplex
