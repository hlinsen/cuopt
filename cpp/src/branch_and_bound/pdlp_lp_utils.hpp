/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <dual_simplex/presolve.hpp>

#include <linear_algebra/sparse_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

// Convert the equality-form simplex problem back to the bounded-row representation preferred by
// PDLP. new_slacks contains exactly one singleton slack column for every row.
template <typename i_t, typename f_t>
inline io::mps_data_model_t<i_t, f_t> simplex_problem_to_mps_data_model(
  const simplex::lp_problem_t<i_t, f_t>& lp,
  const std::vector<i_t>& new_slacks,
  const std::vector<f_t>& simplex_solution,
  std::vector<f_t>& structural_solution)
{
  io::mps_data_model_t<i_t, f_t> mps_model;
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols - static_cast<i_t>(new_slacks.size());
  structural_solution.assign(simplex_solution.begin(), simplex_solution.begin() + n);

  csc_matrix_t<i_t, f_t> A_no_slacks = lp.A;
  std::vector<i_t> cols_to_remove(lp.A.n, 0);
  for (i_t j : new_slacks) {
    cols_to_remove[j] = 1;
  }
  A_no_slacks.remove_columns(cols_to_remove);

  csr_matrix_t<i_t, f_t> csr_A(m, n, 0);
  A_no_slacks.to_compressed_row(csr_A);
  const i_t nz = csr_A.row_start[m];

  mps_model.set_csr_constraint_matrix(
    std::span<const f_t>{csr_A.x.data(), static_cast<size_t>(nz)},
    std::span<const i_t>{csr_A.j.data(), static_cast<size_t>(nz)},
    std::span<const i_t>{csr_A.row_start.data(), static_cast<size_t>(m + 1)});
  mps_model.set_objective_coefficients(
    std::span<const f_t>{lp.objective.data(), static_cast<size_t>(n)});
  mps_model.set_objective_scaling_factor(f_t(1.0));
  mps_model.set_objective_offset(f_t(0.0));
  mps_model.set_variable_lower_bounds(
    std::span<const f_t>{lp.lower.data(), static_cast<size_t>(n)});
  mps_model.set_variable_upper_bounds(
    std::span<const f_t>{lp.upper.data(), static_cast<size_t>(n)});

  std::vector<i_t> slack_map(m, -1);
  for (i_t j : new_slacks) {
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    if (col_end - col_start != 1) { return {}; }
    slack_map[lp.A.i[col_start]] = j;
  }

  std::vector<f_t> constraint_lower(m);
  std::vector<f_t> constraint_upper(m);
  for (i_t i = 0; i < m; ++i) {
    const i_t slack = slack_map[i];
    if (slack < 0) { return {}; }
    const f_t sigma = lp.A.x[lp.A.col_start[slack]];
    if (sigma == f_t(-1.0)) {
      constraint_lower[i] = lp.lower[slack] + lp.rhs[i];
      constraint_upper[i] = lp.upper[slack] + lp.rhs[i];
    } else if (sigma == f_t(1.0)) {
      constraint_lower[i] = -lp.upper[slack] + lp.rhs[i];
      constraint_upper[i] = -lp.lower[slack] + lp.rhs[i];
    } else {
      return {};
    }
  }

  mps_model.set_constraint_lower_bounds(constraint_lower);
  mps_model.set_constraint_upper_bounds(constraint_upper);
  mps_model.set_maximize(false);
  return mps_model;
}

// Expand a bounded-row PDLP point into the equality/slack coordinates expected by MIP cut
// generation. The slack values and reduced costs are reconstructed directly from the simplex
// equations, avoiding assumptions about the sign used for ranged-row slacks.
template <typename i_t, typename f_t>
inline bool expand_pdlp_solution(const simplex::lp_problem_t<i_t, f_t>& lp,
                                 const std::vector<i_t>& new_slacks,
                                 const std::vector<f_t>& structural_x,
                                 const std::vector<f_t>& row_dual,
                                 const std::vector<f_t>& structural_z,
                                 std::vector<f_t>& x,
                                 std::vector<f_t>& y,
                                 std::vector<f_t>& z)
{
  const i_t n = lp.num_cols - static_cast<i_t>(new_slacks.size());
  if (static_cast<i_t>(structural_x.size()) != n || static_cast<i_t>(structural_z.size()) != n ||
      static_cast<i_t>(row_dual.size()) != lp.num_rows) {
    return false;
  }

  x.assign(lp.num_cols, f_t(0.0));
  y = row_dual;
  z.assign(lp.num_cols, f_t(0.0));
  std::copy(structural_x.begin(), structural_x.end(), x.begin());
  std::copy(structural_z.begin(), structural_z.end(), z.begin());

  std::vector<f_t> row_activity(lp.num_rows, f_t(0.0));
  for (i_t j = 0; j < n; ++j) {
    for (i_t p = lp.A.col_start[j]; p < lp.A.col_start[j + 1]; ++p) {
      row_activity[lp.A.i[p]] += lp.A.x[p] * x[j];
    }
  }

  for (i_t j : new_slacks) {
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    if (col_end - col_start != 1) { return false; }
    const i_t i   = lp.A.i[col_start];
    const f_t aij = lp.A.x[col_start];
    if (aij == f_t(0.0)) { return false; }
    x[j] = (lp.rhs[i] - row_activity[i]) / aij;
    z[j] = lp.objective[j] - aij * y[i];
  }

  const auto finite = [](f_t value) { return std::isfinite(value); };
  return std::all_of(x.begin(), x.end(), finite) && std::all_of(y.begin(), y.end(), finite) &&
         std::all_of(z.begin(), z.end(), finite);
}

}  // namespace cuopt::mathematical_optimization::mip
