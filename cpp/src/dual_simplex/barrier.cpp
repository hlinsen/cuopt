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


#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>


namespace cuopt::linear_programming::dual_simplex {

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
      diag(lp.num_cols),
      inv_diag(lp.num_cols),
      inv_sqrt_diag(lp.num_cols),
      AD(lp.num_cols, lp.num_rows, 0),
      DAT(lp.num_rows, lp.num_cols, 0),
      ADAT(lp.num_rows, lp.num_rows, 0),
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
      has_factorization(false)
  {

    // Create the upper bounds vector
    n_upper_bounds = 0;
    for (i_t j = 0; j < lp.num_cols; j++) {
        if (lp.upper[j] < inf) {
            upper_bounds[n_upper_bounds++] = j;
        }
    }
    // Form A*Dinv*A'
    diag.set_scalar(1.0);
    if (n_upper_bounds > 0) {
        for (i_t k = 0; k < n_upper_bounds; k++) {
            i_t j = upper_bounds[k];
            diag[j] = 2.0;
        }
    }
    inv_diag.set_scalar(1.0);
    if (n_upper_bounds > 0) {
        diag.inverse(inv_diag);
    }
    inv_sqrt_diag.set_scalar(1.0);
    if (n_upper_bounds > 0) {
        inv_diag.sqrt(inv_sqrt_diag);
    }

    // Copy A into AD
    AD = lp.A;
    // Perform column scaling on A to get AD^(-1/2)
    AD.scale_columns(inv_sqrt_diag);

    // Form A*Dinv*A'
    float64_t start_form_aat = tic();
    csc_matrix_t<i_t, f_t> DAT(lp.num_cols, lp.num_rows, 0);
    AD.transpose(DAT);
    multiply(AD, DAT, ADAT);
    float64_t aat_time = toc(start_form_aat);
    settings.log.printf("AAT time %.2fs\n", aat_time);
    settings.log.printf("AAT nonzeros %e\n", static_cast<float64_t>(ADAT.col_start[lp.num_rows]));

    if (!settings.use_cudss) {
      chol = std::make_unique<sparse_cholesky_cholmod_t<i_t, f_t>>(lp.num_rows);
    } else {
      chol = std::make_unique<sparse_cholesky_cudss_t<i_t, f_t>>(lp.num_rows);
    }

    // Perform symbolic analysis
    chol->analyze(ADAT);
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

   i_t n_upper_bounds;
   dense_vector_t<i_t, f_t> upper_bounds;
   dense_vector_t<i_t, f_t> c;
   dense_vector_t<i_t, f_t> b;

   dense_vector_t<i_t, f_t> w;
   dense_vector_t<i_t, f_t> x;
   dense_vector_t<i_t, f_t> y;
   dense_vector_t<i_t, f_t> v;
   dense_vector_t<i_t, f_t> z;

   dense_vector_t<i_t, f_t> diag;
   dense_vector_t<i_t, f_t> inv_diag;
   dense_vector_t<i_t, f_t> inv_sqrt_diag;

   csc_matrix_t<i_t, f_t> AD;
   csc_matrix_t<i_t, f_t> DAT;
   csc_matrix_t<i_t, f_t> ADAT;

   std::unique_ptr<sparse_cholesky_base_t<i_t, f_t>> chol;


   bool has_factorization;

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
};


template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::initial_point(iteration_data_t<i_t, f_t>& data)
{
    // Perform a numerical factorization
    data.chol->factorize(data.ADAT);

    // rhs_x <- b
    dense_vector_t<i_t, f_t> rhs_x(lp.rhs);

    dense_vector_t<i_t, f_t> Fu(lp.num_cols);
    data.gather_upper_bounds(lp.upper, Fu);

    dense_vector_t<i_t, f_t> DinvFu(lp.num_cols); // DinvFu = Dinv * Fu
    data.inv_diag.pairwise_product(Fu, DinvFu);

    // rhs_x <-  A * Dinv * F * u  - b
    matrix_vector_multiply(lp.A, 1.0, DinvFu, -1.0, rhs_x);


    // Solve A*Dinv*A'*q = A*Dinv*F*u - b
    dense_vector_t<i_t, f_t> q(lp.num_rows);
    data.chol->solve(rhs_x, q);

    // rhs_x <- A*Dinv*A'*q - rhs_x
    matrix_vector_multiply(data.ADAT, 1.0, q, -1.0, rhs_x);
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
            i_t j = data.upper_bounds[k];
            data.w[k] = lp.upper[j] - data.x[j];
        }
    }

    // Verify A*x = b
    data.primal_residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, data.x, -1.0, data.primal_residual);
    settings.log.printf("||b - A * x||: %e\n", vector_norm2<i_t, f_t>(data.primal_residual));

    if (data.n_upper_bounds > 0) {
        for (i_t k = 0; k < data.n_upper_bounds; k++) {
            i_t j = data.upper_bounds[k];
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
    data.chol->solve(rhs, data.y);

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
void barrier_solver_t<i_t, f_t>::compute_residuals(iteration_data_t<i_t, f_t>& data)
{
    // Compute primal_residual = b - A*x
    data.primal_residual = lp.rhs;
    matrix_vector_multiply(lp.A, -1.0, data.x, 1.0, data.primal_residual);

    // Compute bound_residual = E'*u - w - E'*x
    if (data.n_upper_bounds > 0) {
        for (i_t k = 0; k < data.n_upper_bounds; k++) {
            i_t j = data.upper_bounds[k];
            data.bound_residual[k] = lp.upper[j] - data.w[k] - data.x[j];
        }
    }

    // Compute dual_residual = c - A'*y - z + E*v
    data.c.pairwise_subtract(data.z, data.dual_residual);
    matrix_transpose_vector_multiply(lp.A, -1.0, data.y, 1.0, data.dual_residual);
    if (data.n_upper_bounds > 0) {
        for (i_t k = 0; k < data.n_upper_bounds; k++) {
            i_t j = data.upper_bounds[k];
            data.dual_residual[j] += data.v[k];
        }
    }

    // Compute complementarity_xz_residual = x.*z
    data.x.pairwise_product(data.z, data.complementarity_xz_residual);

    // Compute complementarity_wv_residual = w.*v
    data.w.pairwise_product(data.v, data.complementarity_wv_residual);
}

template <typename i_t, typename f_t>
f_t barrier_solver_t<i_t, f_t>::max_step_to_boundary(const dense_vector_t<i_t, f_t>& x, const dense_vector_t<i_t, f_t>& dx) const
{
    float64_t max_step = 1.0;
    for (i_t i = 0; i < x.size(); i++) {
        if (dx[i] < 0.0) {
            max_step = std::min(max_step, -x[i]/dx[i]);
        }
    }
    return max_step;
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::compute_search_direction(iteration_data_t<i_t, f_t>& data,
                                                          dense_vector_t<i_t, f_t>& dw,
                                                          dense_vector_t<i_t, f_t>& dx,
                                                          dense_vector_t<i_t, f_t>& dy,
                                                          dense_vector_t<i_t, f_t>& dv,
                                                          dense_vector_t<i_t, f_t>& dz)
{
  const bool debug = true;
  // Solves the linear system
  //
  //  dw dx dy dv dz
  // [ 0 A  0   0  0 ] [ dw ] = [ rp  ]
  // [ I E' 0   0  0 ] [ dx ]   [ rw  ]
  // [ 0 0  A' -E  I ] [ dy ]   [ rd  ]
  // [ 0 Z  0   0  X ] [ dv ]   [ rxz ]
  // [ V 0  0   W  0 ] [ dz ]   [ rwv ]

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

  // Form A*D*A'
  if (!data.has_factorization) {
    // copy A into AD
    data.AD = lp.A;
    // Perform column scaling on A to get AD^(1/2)
    data.AD.scale_columns(data.inv_sqrt_diag);
    data.AD.transpose(data.DAT);
    // compute ADAT = AD^(1/2) * (AD^(1/2))^T
    multiply(data.AD, data.DAT, data.ADAT);
    const f_t regularization = 1e-4;
    f_t applied_regularization = 0.0;
    for (i_t j = 0; j < lp.num_rows; j++) {
      const i_t col_start = data.ADAT.col_start[j];
      const i_t col_end   = data.ADAT.col_start[j + 1];
      for (i_t p = col_start; p < col_end; p++) {
        if (data.ADAT.i[p] == j && data.ADAT.x[p] < regularization) {
          data.ADAT.x[p] = regularization;
          applied_regularization += regularization;
        }
      }
    }
    if (applied_regularization > 0.0) {
      settings.log.printf("Applied regularization %.2e\n", applied_regularization);
    }
    // factorize
    data.chol->factorize(data.ADAT);
    data.has_factorization = true;
  }

  // Compute h = primal_rhs + A*inv_diag*(dual_rhs - complementarity_xz_rhs ./ x +
  // E*((complementarity_wv_rhs - v .* bound_rhs) ./ w) )
  dense_vector_t<i_t, f_t> tmp1(data.n_upper_bounds);
  dense_vector_t<i_t, f_t> tmp2(data.n_upper_bounds);
  dense_vector_t<i_t, f_t> tmp3(lp.num_cols);
  dense_vector_t<i_t, f_t> tmp4(lp.num_cols);

  // tmp2 <- v .* bound_rhs
  data.v.pairwise_product(data.bound_rhs, tmp2);
  tmp2.axpy(1.0, data.complementarity_wv_rhs, -1.0);
  tmp2.pairwise_divide(data.w, tmp1);
  // tmp1 <- (complementarity_wv_rhs - v .* bound_rhs) ./ w
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
  dense_vector_t<i_t, f_t> r1 = tmp3;
  // tmp4 <- inv_diag * ( dual_rhs -complementarity_xz_rhs ./ x + E *((complementarity_wv_rhs - v .*
  // bound_rhs) ./ w))
  data.inv_diag.pairwise_product(tmp3, tmp4);

  dense_vector_t<i_t, f_t> h = data.primal_rhs;
  // h <- 1.0 * A * tmp4 + h
  matrix_vector_multiply(lp.A, 1.0, tmp4, 1.0, h);

  // Solve A D^{-1} A^T dy = h
  data.chol->solve(h, dy);

  // y_residual <- ADAT*dy - h
  dense_vector_t<i_t, f_t> y_residual = h;
  matrix_vector_multiply(data.ADAT, 1.0, dy, -1.0, y_residual);
  if (debug) {
    const f_t y_residual_norm = vector_norm_inf<i_t, f_t>(y_residual);
    if (y_residual_norm > 1e-2) {
      settings.log.printf("||ADAT*dy - h|| = %.2e\n", y_residual_norm);
    }
}

  // dx = dinv .* (A'*dy - dual_rhs + complementarity_xz_rhs ./ x  - E *((complementarity_wv_rhs - v
  // .* bound_rhs) ./ w)) r1 <- A'*dy - r1
  matrix_transpose_vector_multiply(lp.A, 1.0, dy, -1.0, r1);
  // dx <- dinv .* r1
  data.inv_diag.pairwise_product(r1, dx);

  // x_residual <- A * dx - primal_rhs
  dense_vector_t<i_t, f_t> x_residual = data.primal_rhs;
  matrix_vector_multiply(lp.A, 1.0, dx, -1.0, x_residual);
  if (debug) {
    const f_t x_residual_norm = vector_norm_inf<i_t, f_t>(x_residual);
    if (x_residual_norm > 1e-2) {
      settings.log.printf("|| A * dx - rp || = %.2e\n", x_residual_norm);
    }
  }

  // dz = (complementarity_xz_rhs - z.* dx) ./ x;
  // tmp3 <- z .* dx
  data.z.pairwise_product(dx, tmp3);
  // tmp3 <- 1.0 * complementarity_xz_rhs - tmp3
  tmp3.axpy(1.0, data.complementarity_xz_rhs, -1.0);
  // dz <- tmp3 ./ x
  tmp3.pairwise_divide(data.x, dz);

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
    if (xz_residual_norm > 1e-2) {
      settings.log.printf("|| Z dx + X dz - rxz || = %.2e\n", xz_residual_norm);
    }
  }

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

  // dw = bound_rhs - E'*dx
  data.gather_upper_bounds(dx, tmp1);
  dw = data.bound_rhs;
  dw.axpy(-1.0, tmp1, 1.0);

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
    if (wv_residual_norm > 1e-2) {
      settings.log.printf("|| V dw + W dv - rwv || = %.2e\n", wv_residual_norm);
    }
  }
}

template <typename i_t, typename f_t>
void barrier_solver_t<i_t, f_t>::solve(const barrier_solver_settings_t<i_t, f_t>& options)
{

    float64_t start_time = tic();

    i_t n = lp.num_cols;
    i_t m = lp.num_rows;

    settings.log.printf("Barrier solver: %d constraints, %d variables, %ld nonzeros\n", m, n, lp.A.col_start[n]);

    // Compute the number of free variables
    i_t num_free_variables = presolve_info.free_variable_pairs.size() / 2;
    settings.log.printf("%d free variables\n", num_free_variables);

    // Compute the number of upper bounds
    i_t num_upper_bounds = 0;
    for (i_t j = 0; j < n; j++) {
        if (lp.upper[j] < inf) {
            num_upper_bounds++;
        }
    }
    iteration_data_t<i_t, f_t> data(lp, num_upper_bounds, settings);
    settings.log.printf("%d finite upper bounds\n", num_upper_bounds);

    initial_point(data);
    compute_residuals(data);

    f_t primal_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.primal_residual),
                                        vector_norm_inf<i_t, f_t>(data.bound_residual));
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(data.dual_residual);
    f_t complementarity_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.complementarity_xz_residual),
                                                 vector_norm_inf<i_t, f_t>(data.complementarity_wv_residual));
    f_t mu = (data.complementarity_xz_residual.sum()  + data.complementarity_wv_residual.sum()) / (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
    settings.log.printf("Initial mu %e\n", mu);
    f_t primal_objective = data.c.inner_product(data.x);

    dense_vector_t<i_t, f_t> restrict_u(num_upper_bounds);
    dense_vector_t<i_t, f_t> upper(lp.upper);
    data.gather_upper_bounds(upper, restrict_u);
    f_t dual_objective = data.b.inner_product(data.y) - restrict_u.inner_product(data.v);

    i_t iter = 0;
    settings.log.printf("         Objective                                  Residual             Step-Length     Time\n");
    settings.log.printf("Iter    Primal               Dual            Primal   Dual    Compl.     Primal Dual     Elapsed\n");
    float64_t elapsed_time = toc(start_time);
    settings.log.printf("%2d   %+.12e %+.12e %.2e %.2e %.2e                   %.1f\n", \
        iter, primal_objective, dual_objective, primal_residual_norm, dual_residual_norm, complementarity_residual_norm, elapsed_time);

    bool converged = primal_residual_norm < options.feasibility_tol && dual_residual_norm < options.optimality_tol && complementarity_residual_norm < options.complementarity_tol;

    while (iter < options.iteration_limit) {
        // Compute the affine step
        data.primal_rhs = data.primal_residual;
        data.bound_rhs = data.bound_residual;
        data.dual_rhs = data.dual_residual;
        data.complementarity_xz_rhs = data.complementarity_xz_residual;
        data.complementarity_wv_rhs = data.complementarity_wv_residual;
        // x.*z ->  -x .* z
        data.complementarity_xz_rhs.multiply_scalar(-1.0);
        // w.*v -> -w .* v
        data.complementarity_wv_rhs.multiply_scalar(-1.0);

        compute_search_direction(data, data.dw_aff, data.dx_aff, data.dy_aff, data.dv_aff, data.dz_aff);

        f_t step_primal_aff = std::min(max_step_to_boundary(data.w, data.dw_aff), max_step_to_boundary(data.x, data.dx_aff));
        f_t step_dual_aff   = std::min(max_step_to_boundary(data.v, data.dv_aff), max_step_to_boundary(data.z, data.dz_aff));

        // w_aff = w + step_primalaff * dw_aff
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
        f_t mu_aff = (complementarity_xz_aff.sum() + complementarity_wv_aff.sum()) / (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
        //settings.log.printf("mu aff %e\n", mu_aff);
        if (mu_aff < 0.0 || mu_aff != mu_aff) {
            settings.log.printf("mu aff bad %e\n", mu_aff);
            exit(1);
        }

        f_t sigma = std::max(0.0, std::min(1.0, std::pow(mu_aff / mu, 3.0)));
        //settings.log.printf("sigma %e\n", sigma);

        // Compute the combined centering corrector step
        data.primal_rhs.set_scalar(0.0);
        data.bound_rhs.set_scalar(0.0);
        data.dual_rhs.set_scalar(0.0);
        // complementarity_xz_rhs = -dx_aff .* dz_aff + sigma * mu
        data.dx_aff.pairwise_product(data.dz_aff, data.complementarity_xz_rhs);
        data.complementarity_xz_rhs.multiply_scalar(-1.0);
        data.complementarity_xz_rhs.add_scalar(sigma * mu);

        // complementarity_wv_rhs = -dw_aff .* dv_aff + sigma * mu
        data.dw_aff.pairwise_product(data.dv_aff, data.complementarity_wv_rhs);
        data.complementarity_wv_rhs.multiply_scalar(-1.0);
        data.complementarity_wv_rhs.add_scalar(sigma * mu);

        compute_search_direction(data, data.dw, data.dx, data.dy, data.dv, data.dz);
        data.has_factorization = false;

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

        f_t step_primal = std::min(max_step_to_boundary(data.w, data.dw), max_step_to_boundary(data.x, data.dx));
        f_t step_dual   = std::min(max_step_to_boundary(data.v, data.dv), max_step_to_boundary(data.z, data.dz));

        data.w.axpy(options.step_scale * step_primal, data.dw, 1.0);
        data.x.axpy(options.step_scale * step_primal, data.dx, 1.0);
        data.y.axpy(options.step_scale * step_dual, data.dy, 1.0);
        data.v.axpy(options.step_scale * step_dual, data.dv, 1.0);
        data.z.axpy(options.step_scale * step_dual, data.dz, 1.0);

        // Handle free variables
        if (num_free_variables > 0)
        {
            for (i_t k = 0; k < 2*num_free_variables; k += 2)
            {
                i_t u = presolve_info.free_variable_pairs[k];
                i_t v = presolve_info.free_variable_pairs[k+1];
                f_t u_val = data.x[u];
                f_t v_val = data.x[v];
                f_t min_val = std::min(u_val, v_val);
                f_t eta = options.step_scale*min_val;
                data.x[u] -= eta;
                data.x[v] -= eta;
            }
        }

        f_t min_w = data.w.minimum();
        f_t min_x = data.x.minimum();
        f_t min_v = data.v.minimum();
        f_t min_z = data.z.minimum();
        if (min_w < 0.0 || min_x < 0.0 || min_v < 0.0 || min_z < 0.0) {
            settings.log.printf("Violated bounds\n");
        }

        compute_residuals(data);

        mu = (data.complementarity_xz_residual.sum()  + data.complementarity_wv_residual.sum()) / (static_cast<f_t>(n) + static_cast<f_t>(num_upper_bounds));
        //settings.log.printf("mu %e\n", mu);

        primal_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.primal_residual),
                                        vector_norm_inf<i_t, f_t>(data.bound_residual));
        dual_residual_norm = vector_norm_inf<i_t, f_t>(data.dual_residual);
        complementarity_residual_norm = std::max(vector_norm_inf<i_t, f_t>(data.complementarity_xz_residual),
                                                 vector_norm_inf<i_t, f_t>(data.complementarity_wv_residual));

        primal_objective = data.c.inner_product(data.x);
        dual_objective = data.b.inner_product(data.y) - restrict_u.inner_product(data.v);
        iter++;
        elapsed_time = toc(start_time);

        settings.log.printf("%2d   %+.12e %+.12e %.2e %.2e %.2e %.2e %.2e %.2e %.1f\n", \
            iter, primal_objective, dual_objective, primal_residual_norm, dual_residual_norm, complementarity_residual_norm, step_primal, step_dual, mu, elapsed_time);

        bool primal_feasible = primal_residual_norm < options.feasibility_tol;
        bool dual_feasible = dual_residual_norm < options.optimality_tol;
        bool small_gap = complementarity_residual_norm < options.complementarity_tol;

        converged = primal_feasible && dual_feasible && small_gap;

        if (converged) {
            settings.log.printf("Optimal solution found\n");
            settings.log.printf("Optimal objective value %.12e with constant %.12e\n", primal_objective, primal_objective + lp.obj_constant);
            settings.log.printf("|| x* ||_inf %e\n", vector_norm_inf<i_t, f_t>(data.x));
            break;
        }
    }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class barrier_solver_t<int, double>;
template class sparse_cholesky_base_t<int, double>;
template class sparse_cholesky_cholmod_t<int, double>;
template class sparse_cholesky_cudss_t<int, double>;
template class iteration_data_t<int, double>;
template class barrier_solver_settings_t<int, double>;
#endif

} // namespace cuopt::linear_programming::dual_simplex
