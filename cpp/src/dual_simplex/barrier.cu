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

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>

#include <numeric>

#include <raft/common/nvtx.hpp>

namespace cuopt::linear_programming::dual_simplex {


template <typename i_t, typename f_t>
class dense_matrix_t {
 public:
  dense_matrix_t(i_t rows, i_t cols)
    : m(rows), n(cols), values(rows * cols, 0.0)
  {}

  void resize(i_t rows, i_t cols)
  {
    m = rows;
    n = cols;
    values.resize(rows * cols, 0.0);
  }

  f_t& operator()(i_t row, i_t col)
  {
    return values[col * m + row];
  }

  f_t operator()(i_t row, i_t col) const
  {
    return values[col * m + row];
  }

  void from_sparse(const csc_matrix_t<i_t, f_t>& A, i_t sparse_column, i_t dense_column)
  {
    for (i_t i = 0; i < m; i++) {
      this->operator()(i, dense_column) = 0.0;
    }

    const i_t col_start = A.col_start[sparse_column];
    const i_t col_end = A.col_start[sparse_column + 1];
    for (i_t p = col_start; p < col_end; p++) {
      this->operator()(A.i[p], dense_column) = A.x[p];
    }
  }

  void add_diagonal(const dense_vector_t<i_t, f_t>& diag)
  {
    for (i_t j = 0; j < n; j++) {
      this->operator()(j, j) += diag[j];
    }
  }

  void set_column(i_t col, const dense_vector_t<i_t, f_t>& x)
  {
    for (i_t i = 0; i < m; i++) {
      this->operator()(i, col) = x[i];
    }
  }

  // y = alpha * A * x + beta * y
  void matrix_vector_multiply(f_t alpha, const dense_vector_t<i_t, f_t>& x, f_t beta, dense_vector_t<i_t, f_t>& y) const
  {
    if (x.size() != n) {
      printf("dense_matrix_t::matrix_vector_multiply: x.size() != n\n");
      exit(1);
    }
    if (y.size() != m) {
      printf("dense_matrix_t::matrix_vector_multiply: y.size() != m\n");
      exit(1);
    }

    for (i_t i = 0; i < m; i++) {
      y[i] *= beta;
    }

    const dense_matrix_t<i_t, f_t>& A = *this;

    for (i_t j = 0; j < n; j++) {
      for (i_t i = 0; i < m; i++) {
        y[i] += alpha * A(i, j) * x[j];
      }
    }
  }

  // y = alpha * A' * x + beta * y
  void transpose_multiply(f_t alpha, const dense_vector_t<i_t, f_t>& x, f_t beta, dense_vector_t<i_t, f_t>& y) const
  {
    if (x.size() != m) {
      printf("dense_matrix_t::transpose_multiply: x.size() != n\n");
      exit(1);
    }
    for (i_t j = 0; j < n; j++) {
      f_t sum = 0.0;
      for (i_t i = 0; i < m; i++) {
        sum += x[i] * this->operator()(i, j);
      }
      y[j] = alpha * sum + beta * y[j];
    }
  }

  // Y <- alpha * A' * X + beta * Y
  void transpose_matrix_multiply(f_t alpha, const dense_matrix_t<i_t, f_t>& X, f_t beta, dense_matrix_t<i_t, f_t>& Y) const
  {
    // X is m x p
    // Y is q x p
    // Y <- alpha * A' * X + beta * Y
    if (X.n != Y.n) {
      printf("dense_matrix_t::transpose_matrix_multiply: X.m != Y.m\n");
      exit(1);
    }
    if (X.m != m) {
      printf("dense_matrix_t::transpose_matrix_multiply: X.m != m\n");
      exit(1);
    }
    for (i_t k = 0; k < X.n; k++) {
      for (i_t j = 0; j < n; j++) {
        f_t sum = 0.0;
        for (i_t i = 0; i < m; i++) {
          sum += this->operator()(i, j) * X(i, k);
        }
        Y(j, k) = alpha * sum + beta * Y(j, k);
      }
    }
  }

  void scale_columns(const dense_vector_t<i_t, f_t>& scale)
  {
    for (i_t j = 0; j < n; j++) {
      for (i_t i = 0; i < m; i++) {
        this->operator()(i, j) *= scale[j];
      }
    }
  }

  void chol(dense_matrix_t<i_t, f_t>& L)
  {
    if (m != n) {
      printf("dense_matrix_t::chol: m != n\n");
      exit(1);
    }
    if (L.m != n) {
      printf("dense_matrix_t::chol: L.m != n\n");
      exit(1);
    }

    // Clear the upper triangular part of L
    for (i_t i = 0; i < n; i++) {
      for (i_t j = i + 1; j < n; j++) {
        L(i, j) = 0.0;
      }
    }

    const dense_matrix_t<i_t, f_t>& A = *this;
    // Compute the cholesky factor and store it in the lower triangular part of L
    for (i_t i = 0; i < n; i++) {
      for (i_t j = 0; j <= i; j++) {
        f_t sum = 0.0;
        for (i_t k = 0; k < j; k++) {
          sum += L(i, k) * L(j, k);
        }

        if (i == j) {
          L(i, j) = sqrt(A(i, i) - sum);
        } else {
          L(i, j) = (1.0 / L(j, j) * (A(i, j) - sum));
        }
      }
    }
  }

  // Assume A = L
  // Solve L * x = b
  void triangular_solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
  {
    if (b.size() != n) {
      printf("dense_matrix_t::triangular_solve: b.size() %d != n %d\n", static_cast<i_t>(b.size()), n);
      exit(1);
    }
    x.resize(n, 0.0);

    // sum_k=0^i-1 L(i, k) * x[k] + L(i, i) * x[i] = b[i]
    // x[i] = (b[i] - sum_k=0^i-1 L(i, k) * x[k]) / L(i, i)
    const dense_matrix_t<i_t, f_t>& L = *this;
    for (i_t i = 0; i < n; i++) {
      f_t sum = 0.0;
      for (i_t k = 0; k < i; k++) {
        sum += L(i, k) * x[k];
      }
      x[i] = (b[i] - sum) / L(i, i);
    }
  }

  // Assume A = L
  // Solve  L^T * x = b
  void triangular_solve_transpose(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x)
  {
    if (b.size() != n) {
      printf("dense_matrix_t::triangular_solve_transpose: b.size() != n\n");
      exit(1);
    }
    x.resize(n, 0.0);

    // L^T = U
    // U * x = b
    // sum_k=i+1^n U(i, k) * x[k] + U(i, i) * x[i] = b[i], i=n-1, n-2, ..., 0
    // sum_k=i+1^n L(k, i) * x[k] + L(i, i) * x[i] = b[i], i=n-1, n-2, ..., 0
    const dense_matrix_t<i_t, f_t>& L = *this;
    for (i_t i = n - 1; i >= 0; i--) {
      f_t sum = 0.0;
      for (i_t k = i + 1; k < n; k++) {
        sum += L(k, i) * x[k];
      }
      x[i] = (b[i] - sum) / L(i, i);
    }
  }

  i_t m;
  i_t n;
  std::vector<f_t> values;
};

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
      A(lp.A),
      primal_residual(lp.num_rows),
      bound_residual(num_upper_bounds),
      dual_residual(lp.num_cols),
      complementarity_xz_residual(lp.num_cols),
      complementarity_wv_residual(num_upper_bounds),
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
      has_factorization(false),
      num_factorizations(0),
      settings_(settings)
  {
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
    inv_sqrt_diag.set_scalar(1.0);
    if (n_upper_bounds > 0) { inv_diag.sqrt(inv_sqrt_diag); }

    n_dense_columns = 0;
    std::vector<i_t> dense_columns_unordered;
    if (settings.eliminate_dense_columns) {
      f_t start_column_density = tic();
      column_density(lp.A, settings, dense_columns_unordered);
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
      dense_columns.clear();
      dense_columns.reserve(n_dense_columns);
      for (i_t j = 0; j < lp.num_cols; j++) {
        if (cols_to_remove[j]) {
          dense_columns.push_back(j);
        }
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
    AD.transpose(AT);

    form_adat();

    chol = std::make_unique<sparse_cholesky_cudss_t<i_t, f_t>>(settings, lp.num_rows);
    chol->set_positive_definite(false);

    // Perform symbolic analysis
    chol->analyze(ADAT);
  }

  void form_adat()
  {
    float64_t start_form_adat = tic();
    const i_t m = AD.m;
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
      inv_diag_prime = inv_diag;
    }

    if (inv_diag_prime.size() != AD.n) {
      settings_.log.printf("inv_diag_prime.size() = %ld, AD.n = %d\n", inv_diag_prime.size(), AD.n);
      exit(1);
    }

    // Scale the columns of A
    AD.scale_columns(inv_diag_prime);

    // Compute A*Dinv*A'
    multiply(AD, AT, ADAT);

    float64_t adat_time = toc(start_form_adat);
    if (num_factorizations == 0) {
      settings_.log.printf("ADAT time %.2fs\n", adat_time);
      settings_.log.printf("ADAT nonzeros %e\n", static_cast<float64_t>(ADAT.col_start[m]));
    }
  }

  i_t solve_adat(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) const
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
      // (U^T * (A_sparse * D_sparse * A_sparse^T)^{-1} U + I) * y = U^T * (A_sparse * D_sparse * A_sparse^T)^{-1} * b
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
      if (solve_status != 0) {
        return solve_status;
      }

      dense_matrix_t<i_t, f_t> AD_dense = A_dense;

      // AD_dense = A_dense * D_dense
      dense_vector_t<i_t, f_t> dense_diag(n_dense_columns);
      i_t k = 0;
      for (i_t j : dense_columns) {
        dense_diag[k++] = std::sqrt(inv_diag[j]);
      }
      AD_dense.scale_columns(dense_diag);

      dense_matrix_t<i_t, f_t> M(AD.m, n_dense_columns);
      dense_matrix_t<i_t, f_t> H(n_dense_columns, n_dense_columns);
      for (i_t k = 0; k < n_dense_columns; k++) {
        dense_vector_t<i_t, f_t> U_col(AD.m);
        // U_col = AD_dense(:, k)
        for (i_t i = 0; i < AD.m; i++) {
          U_col[i] = AD_dense(i, k);
        }
        dense_vector_t<i_t, f_t> M_col(AD.m);
        solve_status = chol->solve(U_col, M_col);
        if (solve_status != 0) {
          return solve_status;
        }
        M.set_column(k, M_col);

        if (debug) {
          dense_vector_t<i_t, f_t> M_residual = U_col;
          matrix_vector_multiply(ADAT, 1.0, M_col, -1.0, M_residual);
          settings_.log.printf("|| A_sparse * D_sparse * A_sparse^T * M(:, k) - AD_dense(:, k) ||_2 = %e\n", vector_norm2<i_t, f_t>(M_residual));
        }
      }
      // A_sparse * D_sparse * A_sparse^T * M = U = AD_dense
      // H = AD_dense^T * M
      AD_dense.transpose_matrix_multiply(1.0, M, 0.0, H);
      dense_vector_t<i_t, f_t> e(n_dense_columns);
      e.set_scalar(1.0);
      // H = AD_dense^T * M + I
      H.add_diagonal(e);

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

      // H = L*L^T
      dense_matrix_t<i_t, f_t> L(n_dense_columns, n_dense_columns);
      H.chol(L);

      dense_vector_t<i_t, f_t> y(n_dense_columns);
      // H *y = g
      // L*L^T * y = g
      // L*u = g
      dense_vector_t<i_t, f_t> u(n_dense_columns);
      L.triangular_solve(g, u);
      // L^T y = u
      L.triangular_solve_transpose(u, y);

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
      if (solve_status != 0) {
        return solve_status;
      }

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

      if (0 && debug)
      {

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

      if (debug)
      {

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
          max_error = std::max(max_error, ij_error);
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
        i_t new_j   = 0;
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
          i_t col_end = error2.col_start[j + 1];
          for (i_t p = col_start; p < col_end; p++) {
          if (std::abs(error2.x[p]) > 1e-6) {
              large_nz++;
              settings_.log.printf("large_nz (%d,%d) %e. m %d\n", error2.i[p], j, error2.x[p], AD.m);
            }
          }
        }

        settings_.log.printf("|| A_sparse * D_sparse * A_sparse^T + A_dense * D_dense * A_dense^T - ADAT ||_1 = %e nz %e large_nz %ld\n", error2.norm1(), static_cast<f_t>(error2.col_start[AD.m]), large_nz);

      }

      if (0 && debug)
      {
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
      if (do_pcg) {
        preconditioned_conjugate_gradient(settings_, b, 1e-9, x);
      }

      return solve_status;
    }
  }

  void to_solution(const lp_problem_t<i_t, f_t>& lp, i_t iterations, f_t objective, f_t user_objective, f_t primal_residual, f_t dual_residual, lp_solution_t<i_t, f_t>& solution)
  {
    solution.x = x;
    solution.y = y;
    dense_vector_t<i_t, f_t> z_tilde(z.size());
    scatter_upper_bounds(v, z_tilde);
    z_tilde.axpy(1.0, z, -1.0);
    solution.z = z_tilde;

    dense_vector_t<i_t, f_t> dual_res = z_tilde;
    dual_res.axpy(-1.0, lp.objective, 1.0);
    matrix_transpose_vector_multiply(lp.A, 1.0, solution.y, 1.0, dual_res);
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_res);
    settings_.log.printf("Solution Dual residual: %e\n", dual_residual_norm);

    solution.iterations = iterations;
    solution.objective = objective;
    solution.user_objective = user_objective;
    solution.l2_primal_residual = primal_residual;
    solution.l2_dual_residual = dual_residual;
  }

  void column_density(const csc_matrix_t<i_t, f_t>& A,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      std::vector<i_t>& columns_to_remove)
  {
    f_t start_column_density = tic();
    const i_t m = A.m;
    const i_t n = A.n;
    if (n < 2) { return; }
    std::vector<f_t> col_nz(n);
    for (i_t j = 0; j < n; j++) {
      const i_t nnz = A.col_start[j + 1] - A.col_start[j];
      col_nz[j]    = static_cast<f_t>(nnz);
    }

    std::vector<i_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(), [&col_nz](i_t i, i_t j) {
      return col_nz[i] < col_nz[j];
    });

    columns_to_remove.reserve(n);

    f_t nz_max = static_cast<f_t>(m) * static_cast<f_t>(m);

    settings.log.printf("Column density time %.2fs\n", toc(start_column_density));

    const f_t threshold = 10 * sqrt(m);
    for (i_t i = 0; i < n; i++) {
      const i_t j         = permutation[i];


      if ( col_nz[j] > threshold && n - i < 100) {
        settings.log.printf("Dense column: nz %10d col %6d in %.2fs\n",
          static_cast<i_t>(col_nz[j]),
          j,
          toc(start_column_density));
        columns_to_remove.push_back(j);
      }
    }
  }

  void scatter_upper_bounds(const dense_vector_t<i_t, f_t>& y, dense_vector_t<i_t, f_t>& z)
  {
    if (n_upper_bounds > 0) {
      for (i_t k = 0; k < n_upper_bounds; k++) {
        i_t j = upper_bounds[k];
        z[j]  = y[k];
      }
    }
  }

  void gather_upper_bounds(const dense_vector_t<i_t, f_t>& z, dense_vector_t<i_t, f_t>& y)
  {
    if (n_upper_bounds > 0) {
      for (i_t k = 0; k < n_upper_bounds; k++) {
        i_t j = upper_bounds[k];
        y[k]  = z[j];
      }
    }
  }

  // v = alpha * A * Dinv * A^T * y + beta * v
  void adat_multiply(f_t alpha, const dense_vector_t<i_t, f_t>& y, f_t beta, dense_vector_t<i_t, f_t>& v) const
  {
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
    dense_vector_t<i_t, f_t> u(n);
    matrix_transpose_vector_multiply(A, 1.0, y, 0.0, u);

    // w = Dinv * u
    dense_vector_t<i_t, f_t> w(n);
    inv_diag.pairwise_product(u, w);

    // y = alpha * A * w + beta * v = alpha * A * Dinv * A^T * y + beta * v
    matrix_vector_multiply(A, alpha, w, beta, v);
  }

  void preconditioned_conjugate_gradient(const simplex_solver_settings_t<i_t, f_t>& settings,
                                         const dense_vector_t<i_t, f_t>& b,
                                         f_t tolerance,
                                         dense_vector_t<i_t, f_t>& xinout) const
  {
    const bool show_pcg_info = false;
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
                          vector_norm_inf<i_t, f_t>(residual));
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
                            vector_norm_inf<i_t, f_t>(residual));
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
      if (show_pcg_info) {
        settings.log.printf("Rejecting PCG solution\n");
      }
    }
  }

  i_t n_upper_bounds;
  dense_vector_t<i_t, f_t> upper_bounds;
  dense_vector_t<i_t, f_t> c;
  dense_vector_t<i_t, f_t> b;

  dense_vector_t<i_t, f_t> w;
  dense_vector_t<i_t, f_t> x;
  dense_vector_t<i_t, f_t> y;
  dense_vector_t<i_t, f_t> v;
  dense_vector_t<i_t, f_t> z;

  dense_vector_t<i_t, f_t> w_save;
  dense_vector_t<i_t, f_t> x_save;
  dense_vector_t<i_t, f_t> y_save;
  dense_vector_t<i_t, f_t> v_save;
  dense_vector_t<i_t, f_t> z_save;

  dense_vector_t<i_t, f_t> diag;
  dense_vector_t<i_t, f_t> inv_diag;
  dense_vector_t<i_t, f_t> inv_sqrt_diag;


  std::vector<f_t> original_A_values;

  csc_matrix_t<i_t, f_t> AD;
  csc_matrix_t<i_t, f_t> AT;
  csc_matrix_t<i_t, f_t> ADAT;

  i_t n_dense_columns;
  std::vector<i_t> dense_columns;
  std::vector<i_t> sparse_mark;
  std::vector<i_t> cols_to_remove;
  dense_matrix_t<i_t, f_t> A_dense;
  const csc_matrix_t<i_t, f_t>& A;

  std::unique_ptr<sparse_cholesky_base_t<i_t, f_t>> chol;

  bool has_factorization;
  i_t num_factorizations;

  dense_vector_t<i_t, f_t> primal_residual;
  dense_vector_t<i_t, f_t> bound_residual;
  dense_vector_t<i_t, f_t> dual_residual;
  dense_vector_t<i_t, f_t> complementarity_xz_residual;
  dense_vector_t<i_t, f_t> complementarity_wv_residual;

  dense_vector_t<i_t, f_t> primal_rhs;
  dense_vector_t<i_t, f_t> bound_rhs;
  dense_vector_t<i_t, f_t> dual_rhs;
  dense_vector_t<i_t, f_t> complementarity_xz_rhs;
  dense_vector_t<i_t, f_t> complementarity_wv_rhs;

  dense_vector_t<i_t, f_t> dw_aff;
  dense_vector_t<i_t, f_t> dx_aff;
  dense_vector_t<i_t, f_t> dy_aff;
  dense_vector_t<i_t, f_t> dv_aff;
  dense_vector_t<i_t, f_t> dz_aff;

  dense_vector_t<i_t, f_t> dw;
  dense_vector_t<i_t, f_t> dx;
  dense_vector_t<i_t, f_t> dy;
  dense_vector_t<i_t, f_t> dv;
  dense_vector_t<i_t, f_t> dz;

  const simplex_solver_settings_t<i_t, f_t>& settings_;
};

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::initial_point(iteration_data_t<i_t, f_t>& data)
{
  // Perform a numerical factorization
  i_t status = data.chol->factorize(data.ADAT);
  if (status != 0) {
    settings.log.printf("Initial factorization failed\n");
    return;
  }
  data.num_factorizations++;

  // rhs_x <- b
  dense_vector_t<i_t, f_t> rhs_x(lp.rhs);

  dense_vector_t<i_t, f_t> Fu(lp.num_cols);
  data.gather_upper_bounds(lp.upper, Fu);

  dense_vector_t<i_t, f_t> DinvFu(lp.num_cols);  // DinvFu = Dinv * Fu
  data.inv_diag.pairwise_product(Fu, DinvFu);

  // rhs_x <-  A * Dinv * F * u  - b
  matrix_vector_multiply(lp.A, 1.0, DinvFu, -1.0, rhs_x);

  // Solve A*Dinv*A'*q = A*Dinv*F*u - b
  dense_vector_t<i_t, f_t> q(lp.num_rows);
  settings.log.printf("||rhs_x|| = %e\n", vector_norm2<i_t, f_t>(rhs_x));
  //i_t solve_status = data.chol->solve(rhs_x, q);
  i_t solve_status = data.solve_adat(rhs_x, q);
  settings.log.printf("Initial solve status %d\n", solve_status);
  settings.log.printf("||q|| = %e\n", vector_norm2<i_t, f_t>(q));

  // rhs_x <- A*Dinv*A'*q - rhs_x
  data.adat_multiply(1.0, q, -1.0, rhs_x);
  //matrix_vector_multiply(data.ADAT, 1.0, q, -1.0, rhs_x);
  settings.log.printf("|| A*Dinv*A'*q - (A*Dinv*F*u - b) || = %e\n", vector_norm2<i_t, f_t>(rhs_x));

  // x = Dinv*(F*u - A'*q)
  data.inv_diag.pairwise_product(Fu, data.x);
  // Fu <- -1.0 * A' * q + 1.0 * Fu
  matrix_transpose_vector_multiply(lp.A, -1.0, q, 1.0, Fu);

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
  matrix_vector_multiply(lp.A, 1.0, data.x, -1.0, data.primal_residual);
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
  matrix_vector_multiply(lp.A, 1.0, Dinvc, 0.0, rhs);

  // Solve A*Dinv*A'*q = A*Dinv*c
  //data.chol->solve(rhs, data.y);
  data.solve_adat(rhs, data.y);

  // z = Dinv*(c - A'*y)
  dense_vector_t<i_t, f_t> cmATy = data.c;
  matrix_transpose_vector_multiply(lp.A, -1.0, data.y, 1.0, cmATy);
  // z <- Dinv * (c - A'*y)
  data.inv_diag.pairwise_product(cmATy, data.z);

  // v = -E'*z
  data.gather_upper_bounds(data.z, data.v);
  data.v.multiply_scalar(-1.0);

  // Verify A'*y + z - E*v = c
  data.z.pairwise_subtract(data.c, data.dual_residual);
  matrix_transpose_vector_multiply(lp.A, 1.0, data.y, 1.0, data.dual_residual);
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
void barrier_solver_t<i_t, f_t>::compute_residuals(const dense_vector_t<i_t, f_t>& w,
                                                   const dense_vector_t<i_t, f_t>& x,
                                                   const dense_vector_t<i_t, f_t>& y,
                                                   const dense_vector_t<i_t, f_t>& v,
                                                   const dense_vector_t<i_t, f_t>& z,
                                                   iteration_data_t<i_t, f_t>& data)
{
  // Compute primal_residual = b - A*x
  data.primal_residual = lp.rhs;
  matrix_vector_multiply(lp.A, -1.0, x, 1.0, data.primal_residual);

  // Compute bound_residual = E'*u - w - E'*x
  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j                  = data.upper_bounds[k];
      data.bound_residual[k] = lp.upper[j] - w[k] - x[j];
    }
  }

  // Compute dual_residual = c - A'*y - z + E*v
  data.c.pairwise_subtract(z, data.dual_residual);
  matrix_transpose_vector_multiply(lp.A, -1.0, y, 1.0, data.dual_residual);
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
  compute_residuals(w, x, y, v, z, data);
  primal_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.primal_residual),
                                  vector_norm_inf<i_t, f_t>(data.bound_residual));
  dual_residual_norm   = vector_norm_inf<i_t, f_t>(data.dual_residual);
  complementarity_residual_norm =
    std::max(vector_norm_inf<i_t, f_t>(data.complementarity_xz_residual),
             vector_norm_inf<i_t, f_t>(data.complementarity_wv_residual));
}

template <typename i_t, typename f_t>
f_t barrier_solver_t<i_t, f_t>::max_step_to_boundary(const dense_vector_t<i_t, f_t>& x,
                                                     const dense_vector_t<i_t, f_t>& dx,
                                                     i_t& index) const
{
  float64_t max_step = 1.0;
  index              = -1;
  for (i_t i = 0; i < x.size(); i++) {
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
f_t barrier_solver_t<i_t, f_t>::max_step_to_boundary(const dense_vector_t<i_t, f_t>& x,
                                                     const dense_vector_t<i_t, f_t>& dx) const
{
  i_t index;
  return max_step_to_boundary(x, dx, index);
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::my_pop_range()
{
  // Else the range will pop from the CPU side while some work is still hapenning on the GPU
  // In benchmarking this is useful, in production we should not sync constantly
  constexpr bool gpu_sync = true;
  if (gpu_sync)
    cudaDeviceSynchronize();
  raft::common::nvtx::pop_range();

}

template <typename i_t, typename f_t>
i_t barrier_solver_t<i_t, f_t>::compute_search_direction(iteration_data_t<i_t, f_t>& data,
                                                         dense_vector_t<i_t, f_t>& dw,
                                                         dense_vector_t<i_t, f_t>& dx,
                                                         dense_vector_t<i_t, f_t>& dy,
                                                         dense_vector_t<i_t, f_t>& dv,
                                                         dense_vector_t<i_t, f_t>& dz,
                                                         f_t& max_residual)
{
  raft::common::nvtx::range fun_scope("Barrier: compute_search_direction");

  const bool debug = true;
  // Solves the linear system
  //
  //  dw dx dy dv dz
  // [ 0 A  0   0  0 ] [ dw ] = [ rp  ]
  // [ I E' 0   0  0 ] [ dx ]   [ rw  ]
  // [ 0 0  A' -E  I ] [ dy ]   [ rd  ]
  // [ 0 Z  0   0  X ] [ dv ]   [ rxz ]
  // [ V 0  0   W  0 ] [ dz ]   [ rwv ]

  raft::common::nvtx::push_range("Barrier: pre A*D*A' formation");

  max_residual = 0.0;
  // diag = z ./ x
  data.z.pairwise_divide(data.x, data.diag);
  // diag = z ./ x + E * (v ./ w) * E'
  if (data.n_upper_bounds > 0) {
    for (i_t k = 0; k < data.n_upper_bounds; k++) {
      i_t j = data.upper_bounds[k];
      data.diag[j] += data.v[k] / data.w[k];
    }
  }

  // inv_diag = 1.0 ./ diag
  data.diag.inverse(data.inv_diag);

  // inv_sqrt_diag <- sqrt(inv_diag)
  data.inv_diag.sqrt(data.inv_sqrt_diag);

  my_pop_range();

  // Form A*D*A'
  if (!data.has_factorization) {
    raft::common::nvtx::range fun_scope("Barrier: CPU ADAT");
    data.form_adat();

    // factorize
    i_t status = data.chol->factorize(data.ADAT);
    if (status < 0) {
      settings.log.printf("Factorization failed.\n");
      return -1;
    }
    data.has_factorization = true;
    data.num_factorizations++;
  }

  raft::common::nvtx::push_range("Barrier: post A*D*A' formation");


  // Compute h = primal_rhs + A*inv_diag*(dual_rhs - complementarity_xz_rhs ./ x +
  // E*((complementarity_wv_rhs - v .* bound_rhs) ./ w) )
  raft::common::nvtx::push_range("Barrier: compute H");

  dense_vector_t<i_t, f_t> tmp1(data.n_upper_bounds);
  dense_vector_t<i_t, f_t> tmp2(data.n_upper_bounds);
  dense_vector_t<i_t, f_t> tmp3(lp.num_cols);
  dense_vector_t<i_t, f_t> tmp4(lp.num_cols);

  // tmp2 <- v .* bound_rhs
  data.v.pairwise_product(data.bound_rhs, tmp2);
  // tmp2 <- complementarity_wv_rhs - v .* bound_rhs
  tmp2.axpy(1.0, data.complementarity_wv_rhs, -1.0);
  // tmp1 <- (complementarity_wv_rhs - v .* bound_rhs) ./ w
  tmp2.pairwise_divide(data.w, tmp1);
  tmp3.set_scalar(0.0);
  // tmp3 <- E * tmp1
  data.scatter_upper_bounds(tmp1, tmp3);

  // tmp4 <- complementarity_xz_rhs ./ x
  data.complementarity_xz_rhs.pairwise_divide(data.x, tmp4);
  // tmp3 <- -complementarity_xz_rhs ./ x + E * ((complementarity_wv_rhs - v .* bound_rhs) ./ w)
  tmp3.axpy(-1.0, tmp4, 1.0);
  // tmp3 <- dual_rhs -complementarity_xz_rhs ./ x + E * ((complementarity_wv_rhs - v .* bound_rhs)
  // ./ w)
  tmp3.axpy(1.0, data.dual_rhs, 1.0);
  // r1 <- dual_rhs -complementarity_xz_rhs ./ x +  E * ((complementarity_wv_rhs - v .* bound_rhs)
  // ./ w)
  dense_vector_t<i_t, f_t> r1       = tmp3;
  dense_vector_t<i_t, f_t> r1_prime = r1;
  // tmp4 <- inv_diag * ( dual_rhs -complementarity_xz_rhs ./ x + E *((complementarity_wv_rhs - v .*
  // bound_rhs) ./ w))
  data.inv_diag.pairwise_product(tmp3, tmp4);

  dense_vector_t<i_t, f_t> h = data.primal_rhs;
  // h <- 1.0 * A * tmp4 + primal_rhs
  matrix_vector_multiply(lp.A, 1.0, tmp4, 1.0, h);

  my_pop_range();

  raft::common::nvtx::push_range("Barrier: Solve A D^{-1} A^T dy = h");

  // Solve A D^{-1} A^T dy = h
  //i_t solve_status = data.chol->solve(h, dy);
  i_t solve_status = data.solve_adat(h, dy);
  if (solve_status < 0) {
    settings.log.printf("Linear solve failed\n");
    return -1;
  }

  my_pop_range();

  // y_residual <- ADAT*dy - h
  raft::common::nvtx::push_range("Barrier: y_residual");
  dense_vector_t<i_t, f_t> y_residual = h;
  //matrix_vector_multiply(data.ADAT, 1.0, dy, -1.0, y_residual);
  data.adat_multiply(1.0, dy, -1.0, y_residual);
  const f_t y_residual_norm = vector_norm_inf<i_t, f_t>(y_residual);
  max_residual              = std::max(max_residual, y_residual_norm);
  if (y_residual_norm > 1e-2) {
    settings.log.printf("|| h || = %.2e\n", vector_norm_inf<i_t, f_t>(h));
    settings.log.printf("||ADAT*dy - h|| = %.2e\n", y_residual_norm);
  }
  if (y_residual_norm > 10) { return -1; }

  if (0 && y_residual_norm > 1e-10) {
    settings.log.printf("Running PCG to improve residual 2-norm %e inf-norm %e\n",
                        vector_norm2<i_t, f_t>(y_residual),
                        y_residual_norm);
    data.preconditioned_conjugate_gradient(
      settings, h, std::min(y_residual_norm / 100.0, 1e-6), dy);
    y_residual = h;
    matrix_vector_multiply(data.ADAT, 1.0, dy, -1.0, y_residual);
    const f_t y_residual_norm_pcg = vector_norm_inf<i_t, f_t>(y_residual);
    settings.log.printf("PCG improved residual || ADAT * dy - h || = %.2e\n", y_residual_norm_pcg);
  }
  my_pop_range();

  // dx = dinv .* (A'*dy - dual_rhs + complementarity_xz_rhs ./ x  - E *((complementarity_wv_rhs - v
  // .* bound_rhs) ./ w))
  raft::common::nvtx::push_range("Barrier: dx formation");
  // r1 <- A'*dy - r1
  matrix_transpose_vector_multiply(lp.A, 1.0, dy, -1.0, r1);
  // dx <- dinv .* r1
  data.inv_diag.pairwise_product(r1, dx);

  // dx_residual <- D * dx - A'*dy - r1
  dense_vector_t<i_t, f_t> dx_residual(lp.num_cols);
  // dx_residual <- D*dx
  data.diag.pairwise_product(dx, dx_residual);
  // dx_residual <- -A'*dy + D*dx
  matrix_transpose_vector_multiply(lp.A, -1.0, dy, 1.0, dx_residual);
  // dx_residual <- D*dx - A'*dy + r1
  dx_residual.axpy(1.0, r1_prime, 1.0);
  if (debug) {
    const f_t dx_residual_norm = vector_norm_inf<i_t, f_t>(dx_residual);
    max_residual               = std::max(max_residual, dx_residual_norm);
    if (dx_residual_norm > 1e-2) {
      settings.log.printf("|| D * dx - A'*y + r1 || = %.2e\n", dx_residual_norm);
    }
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dx_residual_2");
  dense_vector_t<i_t, f_t> dx_residual_2(lp.num_cols);
  // dx_residual_2 <- D^-1 * (A'*dy - r1)
  data.inv_diag.pairwise_product(r1, dx_residual_2);
  // dx_residual_2 <- D^-1 * (A'*dy - r1) - dx
  dx_residual_2.axpy(-1.0, dx, 1.0);
  // dx_residual_2 <- D^-1 * (A'*dy - r1) - dx
  const f_t dx_residual_2_norm = vector_norm_inf<i_t, f_t>(dx_residual_2);
  max_residual                 = std::max(max_residual, dx_residual_2_norm);
  if (dx_residual_2_norm > 1e-2) {
    settings.log.printf("|| D^-1 (A'*dy - r1) - dx || = %.2e\n", dx_residual_2_norm);
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dx_residual_5_6");
  dense_vector_t<i_t, f_t> dx_residual_5(lp.num_cols);
  dense_vector_t<i_t, f_t> dx_residual_6(lp.num_rows);
  // dx_residual_5 <- D^-1 * (A'*dy - r1)
  data.inv_diag.pairwise_product(r1, dx_residual_5);
  // dx_residual_6 <- A * D^-1 (A'*dy - r1)
  matrix_vector_multiply(lp.A, 1.0, dx_residual_5, 0.0, dx_residual_6);
  // dx_residual_6 <- A * D^-1 (A'*dy - r1) - A * dx
  matrix_vector_multiply(lp.A, -1.0, dx, 1.0, dx_residual_6);
  const f_t dx_residual_6_norm = vector_norm_inf<i_t, f_t>(dx_residual_6);
  max_residual                 = std::max(max_residual, dx_residual_6_norm);
  if (dx_residual_6_norm > 1e-2) {
    settings.log.printf("|| A * D^-1 (A'*dy - r1) - A * dx || = %.2e\n", dx_residual_6_norm);
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dx_residual_3_4");

  dense_vector_t<i_t, f_t> dx_residual_3(lp.num_cols);
  // dx_residual_3 <- D^-1 * r1
  data.inv_diag.pairwise_product(r1_prime, dx_residual_3);
  dense_vector_t<i_t, f_t> dx_residual_4(lp.num_rows);
  // dx_residual_4 <- A * D^-1 * r1
  matrix_vector_multiply(lp.A, 1.0, dx_residual_3, 0.0, dx_residual_4);
  // dx_residual_4 <-  A * D^-1 * r1 + A * dx
  matrix_vector_multiply(lp.A, 1.0, dx, 1.0, dx_residual_4);
  // dx_residual_4 <- - A * D^-1 * r1 - A * dx + ADAT * dy
  my_pop_range();

#if CHECK_FORM_ADAT
  csc_matrix_t<i_t, f_t> ADinv = lp.A;
  ADinv.scale_columns(data.inv_diag);
  csc_matrix_t<i_t, f_t> ADinvAT(lp.num_rows, lp.num_rows, 1);
  csc_matrix_t<i_t, f_t> Atranspose(1, 1, 0);
  lp.A.transpose(Atranspose);
  multiply(ADinv, Atranspose, ADinvAT);
  matrix_vector_multiply(ADinvAT, 1.0, dy, -1.0, dx_residual_4);
  const f_t dx_residual_4_norm = vector_norm_inf<i_t, f_t>(dx_residual_4);
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

  raft::common::nvtx::push_range("Barrier: dx_residual_7");
  dense_vector_t<i_t, f_t> dx_residual_7 = h;
  //matrix_vector_multiply(data.ADAT, 1.0, dy, -1.0, dx_residual_7);
  data.adat_multiply(1.0, dy, -1.0, dx_residual_7);
  const f_t dx_residual_7_norm = vector_norm_inf<i_t, f_t>(dx_residual_7);
  max_residual                 = std::max(max_residual, dx_residual_7_norm);
  if (dx_residual_7_norm > 1e-2) {
    settings.log.printf("|| A D^{-1} A^T * dy - h || = %.2e\n", dx_residual_7_norm);
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: x_residual");
  // x_residual <- A * dx - primal_rhs
  dense_vector_t<i_t, f_t> x_residual = data.primal_rhs;
  matrix_vector_multiply(lp.A, 1.0, dx, -1.0, x_residual);
  if (debug) {
    const f_t x_residual_norm = vector_norm_inf<i_t, f_t>(x_residual);
    max_residual              = std::max(max_residual, x_residual_norm);
    if (x_residual_norm > 1e-2) {
      settings.log.printf("|| A * dx - rp || = %.2e\n", x_residual_norm);
    }
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dz formation");
  // dz = (complementarity_xz_rhs - z.* dx) ./ x;
  // tmp3 <- z .* dx
  data.z.pairwise_product(dx, tmp3);
  // tmp3 <- 1.0 * complementarity_xz_rhs - tmp3
  tmp3.axpy(1.0, data.complementarity_xz_rhs, -1.0);
  // dz <- tmp3 ./ x
  tmp3.pairwise_divide(data.x, dz);
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: xz_residual");
  // xz_residual <- z .* dx + x .* dz - complementarity_xz_rhs
  dense_vector_t<i_t, f_t> xz_residual = data.complementarity_xz_rhs;
  dense_vector_t<i_t, f_t> zdx(lp.num_cols);
  dense_vector_t<i_t, f_t> xdz(lp.num_cols);
  data.z.pairwise_product(dx, zdx);
  data.x.pairwise_product(dz, xdz);
  xz_residual.axpy(1.0, zdx, -1.0);
  xz_residual.axpy(1.0, xdz, 1.0);
  if (debug) {
    const f_t xz_residual_norm = vector_norm_inf<i_t, f_t>(xz_residual);
    max_residual               = std::max(max_residual, xz_residual_norm);
    if (xz_residual_norm > 1e-2) {
      settings.log.printf("|| Z dx + X dz - rxz || = %.2e\n", xz_residual_norm);
    }
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dv formation");
  // dv = (v .* E' dx + complementarity_wv_rhs - v .* bound_rhs) ./ w
  // tmp1 <- E' * dx
  data.gather_upper_bounds(dx, tmp1);
  // tmp2 <- v .* E' * dx
  data.v.pairwise_product(tmp1, tmp2);
  // tmp1 <- v .* bound_rhs
  data.v.pairwise_product(data.bound_rhs, tmp1);
  // tmp1 <- v .* E' * dx - v . * bound_rhs
  tmp1.axpy(1.0, tmp2, -1.0);
  // tmp1 <- v .* E' * dx + complementarity_wv_rhs - v.* bound_rhs
  tmp1.axpy(1.0, data.complementarity_wv_rhs, 1.0);
  // dv <- tmp1 ./ w
  tmp1.pairwise_divide(data.w, dv);
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dv_residual");
  dense_vector_t<i_t, f_t> dv_residual(data.n_upper_bounds);
  dense_vector_t<i_t, f_t> dv_residual_2(data.n_upper_bounds);
  // dv_residual <- E'*dx
  data.gather_upper_bounds(dx, dv_residual);
  // dv_residual_2 <- v .* E' * dx
  data.v.pairwise_product(dv_residual, dv_residual_2);
  // dv_residual_2 <- -v .* E' * dx
  dv_residual_2.multiply_scalar(-1.0);
  // dv_residual_ <- W .* dv
  data.w.pairwise_product(dv, dv_residual);
  // dv_residual <- -v .* E' * dx + w .* dv
  dv_residual.axpy(1.0, dv_residual_2, 1.0);
  // dv_residual <- -v .* E' * dx + w .* dv - complementarity_wv_rhs
  dv_residual.axpy(-1.0, data.complementarity_wv_rhs, 1.0);
  // dv_residual_2 <- V * bound_rhs
  data.v.pairwise_product(data.bound_rhs, dv_residual_2);
  // dv_residual <- -v .* E' * dx + w .* dv - complementarity_wv_rhs + v .* bound_rhs
  dv_residual.axpy(1.0, dv_residual_2, 1.0);
  if (debug) {
    const f_t dv_residual_norm = vector_norm_inf<i_t, f_t>(dv_residual);
    max_residual               = std::max(max_residual, dv_residual_norm);
    if (dv_residual_norm > 1e-2) {
      settings.log.printf(
        "|| -v .* E' * dx + w .* dv - complementarity_wv_rhs - v .* bound_rhs || = %.2e\n",
        dv_residual_norm);
    }
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dual_residual");
  // dual_residual <- A' * dy - E * dv  + dz -  dual_rhs
  dense_vector_t<i_t, f_t> dual_residual(lp.num_cols);
  // dual_residual <- E * dv
  data.scatter_upper_bounds(dv, dual_residual);
  // dual_residual <- A' * dy - E * dv
  matrix_transpose_vector_multiply(lp.A, 1.0, dy, -1.0, dual_residual);
  // dual_residual <- A' * dy - E * dv + dz
  dual_residual.axpy(1.0, dz, 1.0);
  // dual_residual <- A' * dy - E * dv + dz - dual_rhs
  dual_residual.axpy(-1.0, data.dual_rhs, 1.0);
  if (debug) {
    const f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_residual);
    max_residual                 = std::max(max_residual, dual_residual_norm);
    if (dual_residual_norm > 1e-2) {
      settings.log.printf("|| A' * dy - E * dv  + dz -  dual_rhs || = %.2e\n", dual_residual_norm);
    }
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: dw formation");
  // dw = bound_rhs - E'*dx
  data.gather_upper_bounds(dx, tmp1);
  dw = data.bound_rhs;
  dw.axpy(-1.0, tmp1, 1.0);
  // dw_residual <- dw + E'*dx - bound_rhs
  dense_vector_t<i_t, f_t> dw_residual(data.n_upper_bounds);
  data.gather_upper_bounds(dx, dw_residual);
  dw_residual.axpy(1.0, dw, 1.0);
  dw_residual.axpy(-1.0, data.bound_rhs, 1.0);
  if (debug) {
    const f_t dw_residual_norm = vector_norm_inf<i_t, f_t>(dw_residual);
    max_residual               = std::max(max_residual, dw_residual_norm);
    if (dw_residual_norm > 1e-2) {
      settings.log.printf("|| dw + E'*dx - bound_rhs || = %.2e\n", dw_residual_norm);
    }
  }
  my_pop_range();

  raft::common::nvtx::push_range("Barrier: wv_residual");
  // wv_residual <- v .* dw + w .* dv - complementarity_wv_rhs
  dense_vector_t<i_t, f_t> wv_residual = data.complementarity_wv_rhs;
  dense_vector_t<i_t, f_t> vdw(data.n_upper_bounds);
  dense_vector_t<i_t, f_t> wdv(data.n_upper_bounds);
  data.v.pairwise_product(dw, vdw);
  data.w.pairwise_product(dv, wdv);
  wv_residual.axpy(1.0, vdw, -1.0);
  wv_residual.axpy(1.0, wdv, 1.0);
  if (debug) {
    const f_t wv_residual_norm = vector_norm_inf<i_t, f_t>(wv_residual);
    max_residual               = std::max(max_residual, wv_residual_norm);
    if (wv_residual_norm > 1e-2) {
      settings.log.printf("|| V dw + W dv - rwv || = %.2e\n", wv_residual_norm);
    }
  }
  my_pop_range();

  my_pop_range(); // Pop the post A*D*A' formation range

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
    if (relative_primal_residual < 100 * options.feasibility_tol &&
        relative_dual_residual < 100 * options.optimality_tol &&
        relative_complementarity_residual < 100 * options.complementarity_tol) {
      settings.log.printf("Suboptimal solution found\n");
      data.to_solution(lp,
        iter,
                       primal_objective,
                       primal_objective + lp.obj_constant,
                       vector_norm2<i_t, f_t>(data.primal_residual),
                       vector_norm2<i_t, f_t>(data.dual_residual),
                       solution);
      return lp_status_t::OPTIMAL; // TODO: Barrier should probably have a separate suboptimal status
    }
    f_t primal_residual_norm;
    f_t dual_residual_norm;
    f_t complementarity_residual_norm;
    compute_residual_norms(data.w_save,
                           data.x_save,
                           data.y_save,
                           data.v_save,
                           data.z_save,
                           data,
                           primal_residual_norm,
                           dual_residual_norm,
                           complementarity_residual_norm);
    primal_objective         = data.c.inner_product(data.x_save);
    relative_primal_residual = primal_residual_norm / (1.0 + norm_b);
    relative_dual_residual   = dual_residual_norm / (1.0 + norm_c);
    relative_complementarity_residual =
      complementarity_residual_norm / (1.0 + std::abs(primal_objective));

    if (relative_primal_residual < 100 * options.feasibility_tol &&
        relative_dual_residual < 100 * options.optimality_tol &&
        relative_complementarity_residual < 100 * options.complementarity_tol) {
      settings.log.printf(
        "Restoring previous solution: primal %.2e dual %.2e complementarity %.2e\n",
        relative_primal_residual,
        relative_dual_residual,
        relative_complementarity_residual);
      data.to_solution(lp, iter,
                       primal_objective,
                       primal_objective + lp.obj_constant,
                       vector_norm2<i_t, f_t>(data.primal_residual),
                       vector_norm2<i_t, f_t>(data.dual_residual),
                       solution);
      settings.log.printf("Suboptimal solution found in %d iterations and %.2f seconds\n", iter, toc(start_time));
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
      return lp_status_t::OPTIMAL; // TODO: Barrier should probably have a separate suboptimal status
    } else {
      settings.log.printf(
        "Primal residual %.2e dual residual %.2e complementarity residual %.2e\n",
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
  lp_status_t status = lp_status_t::UNSET;

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
  compute_residuals(data.w, data.x, data.y, data.v, data.z, data);

  f_t primal_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.primal_residual),
                                      vector_norm_inf<i_t, f_t>(data.bound_residual));
  f_t dual_residual_norm   = vector_norm_inf<i_t, f_t>(data.dual_residual);
  f_t complementarity_residual_norm =
    std::max(vector_norm_inf<i_t, f_t>(data.complementarity_xz_residual),
             vector_norm_inf<i_t, f_t>(data.complementarity_wv_residual));
  f_t mu = (data.complementarity_xz_residual.sum() + data.complementarity_wv_residual.sum()) /
           (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));

  f_t norm_b = vector_norm_inf<i_t, f_t>(data.b);
  f_t norm_c = vector_norm_inf<i_t, f_t>(data.c);

  f_t primal_objective = data.c.inner_product(data.x);

  f_t relative_primal_residual = primal_residual_norm / (1.0 + norm_b);
  f_t relative_dual_residual   = dual_residual_norm / (1.0 + norm_c);
  f_t relative_complementarity_residual =
    complementarity_residual_norm / (1.0 + std::abs(primal_objective));

  dense_vector_t<i_t, f_t> restrict_u(num_upper_bounds);
  dense_vector_t<i_t, f_t> upper(lp.upper);
  data.gather_upper_bounds(upper, restrict_u);
  f_t dual_objective = data.b.inner_product(data.y) - restrict_u.inner_product(data.v);

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

  data.w_save = data.w;
  data.x_save = data.x;
  data.y_save = data.y;
  data.v_save = data.v;
  data.z_save = data.z;

  const i_t iteration_limit = std::min(options.iteration_limit, settings.iteration_limit);

  while (iter < iteration_limit) {
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
    data.primal_rhs             = data.primal_residual;
    data.bound_rhs              = data.bound_residual;
    data.dual_rhs               = data.dual_residual;
    data.complementarity_xz_rhs = data.complementarity_xz_residual;
    data.complementarity_wv_rhs = data.complementarity_wv_residual;
    // x.*z ->  -x .* z
    data.complementarity_xz_rhs.multiply_scalar(-1.0);
    // w.*v -> -w .* v
    data.complementarity_wv_rhs.multiply_scalar(-1.0);

    f_t max_affine_residual = 0.0;
    i_t status              = compute_search_direction(
      data, data.dw_aff, data.dx_aff, data.dy_aff, data.dv_aff, data.dz_aff, max_affine_residual);
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

    // Compute the combined centering corrector step
    data.primal_rhs.set_scalar(0.0);
    data.bound_rhs.set_scalar(0.0);
    data.dual_rhs.set_scalar(0.0);
    // complementarity_xz_rhs = -dx_aff .* dz_aff + sigma * mu
    data.dx_aff.pairwise_product(data.dz_aff, data.complementarity_xz_rhs);
    data.complementarity_xz_rhs.multiply_scalar(-1.0);
    data.complementarity_xz_rhs.add_scalar(new_mu);

    // complementarity_wv_rhs = -dw_aff .* dv_aff + sigma * mu
    data.dw_aff.pairwise_product(data.dv_aff, data.complementarity_wv_rhs);
    data.complementarity_wv_rhs.multiply_scalar(-1.0);
    data.complementarity_wv_rhs.add_scalar(new_mu);

    f_t max_corrector_residual = 0.0;
    status                     = compute_search_direction(
      data, data.dw, data.dx, data.dy, data.dv, data.dz, max_corrector_residual);
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
    if (toc(start_time) > settings.time_limit) {
      settings.log.printf("Barrier time limit exceeded\n");
      return lp_status_t::TIME_LIMIT;
    }
    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      settings.log.printf("Barrier solver halted\n");
      return lp_status_t::CONCURRENT_LIMIT;
    }

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

    f_t step_primal = options.step_scale * max_step_primal;
    f_t step_dual   = options.step_scale * max_step_dual;

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

    compute_residual_norms(data.w,
                           data.x,
                           data.y,
                           data.v,
                           data.z,
                           data,
                           primal_residual_norm,
                           dual_residual_norm,
                           complementarity_residual_norm);

    mu = (data.complementarity_xz_residual.sum() + data.complementarity_wv_residual.sum()) /
         (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));

    primal_objective = data.c.inner_product(data.x);
    dual_objective   = data.b.inner_product(data.y) - restrict_u.inner_product(data.v);

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
                       solution);
      return lp_status_t::OPTIMAL;
    }
  }
  data.to_solution(lp, iter,
    primal_objective,
    primal_objective + lp.obj_constant,
    vector_norm2<i_t, f_t>(data.primal_residual),
    vector_norm2<i_t, f_t>(data.dual_residual),
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
