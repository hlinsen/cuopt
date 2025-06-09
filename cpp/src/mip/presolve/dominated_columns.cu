/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "dominated_columns.cuh"

#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
dominated_columns_t<i_t, f_t>::dominated_columns_t(problem_t<i_t, f_t>& problem_)
  : problem(problem_), stream(problem.handle_ptr->get_stream())
{
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::identify_candidate_variables(
  typename problem_t<i_t, f_t>::host_view_t& host_problem,
  bound_presolve_t<i_t, f_t>& bounds_presolve)
{
  bounds_presolve.solve(problem,
                        cuopt::make_span(problem.variable_lower_bounds),
                        cuopt::make_span(problem.variable_upper_bounds));
  auto lb_bars = cuopt::host_copy(bounds_presolve.upd.lb, stream);
  auto ub_bars = cuopt::host_copy(bounds_presolve.upd.ub, stream);
  // candidates.resize(problem.n_variables);
  for (int i = 0; i < problem.n_variables; ++i) {
    f_t lb_bar      = lb_bars[i];
    f_t ub_bar      = ub_bars[i];
    f_t lb_original = host_problem.variable_lower_bounds[i];
    f_t ub_original = host_problem.variable_upper_bounds[i];
    // strenghtened bounds are included in original bounds means free
    // One of the bounds is infinite we can apply theorem 1.
    if (lb_bar >= lb_original && ub_bar <= ub_original &&
        (lb_bar == -std::numeric_limits<f_t>::infinity() ||
         ub_bar == std::numeric_limits<f_t>::infinity())) {
      candidates.push_back(host_problem.variables[i]);
    }
  }
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::compute_signatures(
  typename problem_t<i_t, f_t>::host_view_t& host_problem)
{
  for (int i = 0; i < problem.n_constraints; ++i) {
    auto row_offset = host_problem.offsets[i];
    auto nnz_in_row = host_problem.offsets[i + 1] - row_offset;

    for (int j = 0; j < nnz_in_row; ++j) {
      auto col = host_problem.variables[row_offset + j];
      signatures[col].set(i % signature_size);
    }
  }
}

template <typename i_t, typename f_t>
std::unordered_map<i_t, std::pair<i_t, i_t>> dominated_columns_t<i_t, f_t>::find_shortest_rows(
  typename problem_t<i_t, f_t>::host_view_t& host_problem)
{
  std::unordered_map<i_t, std::pair<i_t, i_t>> shortest_rows;
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto col           = candidates[i];
    auto col_offset    = host_problem.reverse_offsets[col];
    auto nnz_in_col    = host_problem.reverse_offsets[col + 1] - col_offset;
    shortest_rows[col] = {std::numeric_limits<i_t>::max(), -1};
    for (int j = 0; j < nnz_in_col; ++j) {
      auto row         = host_problem.reverse_constraints[col_offset + j];
      auto row_size    = host_problem.offsets[row + 1] - host_problem.offsets[row];
      auto coefficient = host_problem.reverse_coefficients[col_offset + j];
      // Check for LesserThanOrEqual
      if ((host_problem.constraint_lower_bounds[row] == -std::numeric_limits<f_t>::max() &&
           host_problem.constraint_upper_bounds[row] != std::numeric_limits<f_t>::max())) {
        if (coefficient > 0) {
          auto [min_row_size, row_id] = shortest_rows[col];
          if (row_size < min_row_size) { shortest_rows[col] = std::make_pair(row_size, row); }
        }
      }

      // Check for equality
      if (problem.integer_equal(host_problem.constraint_lower_bounds[row],
                                host_problem.constraint_upper_bounds[row])) {
        auto [min_row_size, row_id] = shortest_rows[col];
        if (row_size < min_row_size) { shortest_rows[col] = std::make_pair(row_size, row); }
      }
    }
  }
  return shortest_rows;
}

template <typename i_t, typename f_t>
bool dominated_columns_t<i_t, f_t>::dominates(
  typename problem_t<i_t, f_t>::host_view_t& host_problem, i_t col1, i_t col2, i_t row)
{
  if (!(signatures[col1][row] && signatures[col2][row])) { return false; }

  // Get column vectors for comparison
  auto col1_offset = host_problem.reverse_offsets[col1];
  auto col1_nnz    = host_problem.reverse_offsets[col1 + 1] - col1_offset;
  auto col2_offset = host_problem.reverse_offsets[col2];
  auto col2_nnz    = host_problem.reverse_offsets[col2 + 1] - col2_offset;

  // Check objective coefficients (i)
  if (host_problem.objective_coefficients[col1] > host_problem.objective_coefficients[col2]) {
    return false;
  }

  // Check constraint coefficients (ii)
  for (int i = 0; i < col1_nnz; ++i) {
    auto row1   = host_problem.reverse_constraints[col1_offset + i];
    auto coeff1 = host_problem.reverse_coefficients[col1_offset + i];

    auto coeff2 = 0;
    for (int j = 0; j < col2_nnz; ++j) {
      auto row2 = host_problem.reverse_constraints[col2_offset + j];
      if (row1 == row2) {
        coeff2 = host_problem.reverse_coefficients[col2_offset + j];
        break;
      }
    }
    if (coeff1 > coeff2) { return false; }
  }

  // Check variable types (iii)
  bool col1_is_int = host_problem.variable_types[col1] == var_t::INTEGER;
  bool col2_is_int = host_problem.variable_types[col2] == var_t::INTEGER;
  if (!(col1_is_int || !col2_is_int)) { return false; }

  return true;
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::update_variable_bounds(
  typename problem_t<i_t, f_t>::host_view_t& host_problem, i_t col1, i_t col2)
{
  // Get bounds for both variables
  f_t l1 = host_problem.variable_lower_bounds[col1];
  f_t u1 = host_problem.variable_upper_bounds[col1];
  f_t l2 = host_problem.variable_lower_bounds[col2];
  f_t u2 = host_problem.variable_upper_bounds[col2];

  if (u1 == std::numeric_limits<f_t>::infinity()) {
    // case i: xk can be set to lk
    host_problem.variable_upper_bounds[col2] = l2;
    host_problem.variable_lower_bounds[col1] = l2;
  } else if (l2 == -std::numeric_limits<f_t>::infinity()) {
    // case iii: xk can be set to uk
    host_problem.variable_lower_bounds[col1] = u2;
    host_problem.variable_upper_bounds[col2] = u2;
  }
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::presolve(bound_presolve_t<i_t, f_t>& bounds_presolve)
{
  auto host_problem = problem.to_host();
  identify_candidate_variables(host_problem, bounds_presolve);
  compute_signatures(host_problem);
  auto shortest_rows = find_shortest_rows(host_problem);
  for (const auto& [cand, pair] : shortest_rows) {
    auto const& [row_size, row] = pair;
    auto row_offset             = host_problem.offsets[row];
    auto nnz_in_row             = host_problem.offsets[row + 1] - row_offset;
    // All the variables in this row are the candidates to be dominated by cand
    for (int j = 0; j < nnz_in_row; ++j) {
      auto col = host_problem.variables[row_offset + j];
      if (dominates(host_problem, cand, col, row)) {
        update_variable_bounds(host_problem, cand, col);
      }
    }
  }
}

}  // namespace cuopt::linear_programming::detail

template struct cuopt::linear_programming::detail::dominated_columns_t<int, double>;
