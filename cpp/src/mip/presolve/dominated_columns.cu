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

#include <cmath>
#include <utilities/copy_helpers.hpp>
#include "trivial_presolve.cuh"

auto constexpr const COEFF_EPSILON = 1e-9;

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
dominated_columns_t<i_t, f_t>::dominated_columns_t(problem_t<i_t, f_t>& problem_)
  : problem(problem_), stream(problem.handle_ptr->get_stream())
{
}

template <typename i_t, typename f_t>
std::vector<i_t> dominated_columns_t<i_t, f_t>::identify_candidate_variables(
  typename problem_t<i_t, f_t>::host_view_t& host_problem,
  bound_presolve_t<i_t, f_t>& bounds_presolve,
  std::vector<f_t> const& lb_bars,
  std::vector<f_t> const& ub_bars)
{
  // std::cout << "Identifying candidate variables" << std::endl;
  auto lb = cuopt::host_copy(problem.variable_lower_bounds, stream);
  auto ub = cuopt::host_copy(problem.variable_upper_bounds, stream);
  std::vector<i_t> candidates;
  // auto changed_variables = cuopt::host_copy(bounds_presolve.upd.changed_variables, stream);
  // Tolerance for determining significant bound changes
  auto variable_names   = problem.original_problem_ptr->get_variable_names();
  auto variable_mapping = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  for (int i = 0; i < problem.n_variables; ++i) {
    // if (candidates.size() > 10) { break; }
    f_t lb_bar      = lb_bars[i];
    f_t ub_bar      = ub_bars[i];
    f_t original_lb = host_problem.original_variable_lower_bounds[i];
    f_t original_ub = host_problem.original_variable_upper_bounds[i];

    auto is_lower_implied = lb_bar >= original_lb - problem.tolerances.absolute_tolerance;
    auto is_upper_implied = ub_bar <= original_ub + problem.tolerances.absolute_tolerance;

    // if (i == 0) {
    // std::cout << "col: " << i << " (" << variable_names[i] << ")" << std::endl;
    // std::cout << "lb_updated: " << lb_updated << ", ub_updated: " << ub_updated << std::endl;
    // std::cout << "Variable " << i << " has bounds " << original_lb << " " << original_ub
    //           << " and strengthened bounds " << lb_bar << " " << ub_bar << std::endl;
    // std::cout << "Implied free: " << implied_free << std::endl;
    // }
    if (is_lower_implied || is_upper_implied) {
      std::cout << "col " << i << " (" << variable_names[variable_mapping[i]]
                << ") is lower implied: " << is_lower_implied
                << " is upper implied: " << is_upper_implied << std::endl;
      candidates.push_back(i);
    }
    // One of the bounds is infinite we can apply theorem 1.
    else if (lb[i] == -std::numeric_limits<f_t>::infinity() ||
             ub[i] == std::numeric_limits<f_t>::infinity()) {
      candidates.push_back(i);
    }
  }
  // std::cout << "Candidate variables identified" << std::endl;
  return candidates;
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::compute_signatures(
  typename problem_t<i_t, f_t>::host_view_t& host_problem)
{
  // std::cout << "Computing signatures" << std::endl;
  signatures.resize(problem.n_variables);
  for (int i = 0; i < problem.n_constraints; ++i) {
    // std::cout << "Computing signature for constraint " << i << std::endl;
    auto row_offset = host_problem.offsets[i];
    auto nnz_in_row = host_problem.offsets[i + 1] - row_offset;
    // std::cout << "NNZ in row " << i << " is " << nnz_in_row << std::endl;
    for (int j = 0; j < nnz_in_row; ++j) {
      auto col = host_problem.variables[row_offset + j];
      // std::cout << "Setting signature for variable " << col << std::endl;
      signatures[col].set(i % signature_size);
    }
  }
  // std::cout << "Signatures computed" << std::endl;
}

template <typename i_t, typename f_t>
std::map<i_t, std::pair<i_t, i_t>> dominated_columns_t<i_t, f_t>::find_shortest_rows(
  typename problem_t<i_t, f_t>::host_view_t& host_problem, std::vector<i_t> const& candidates)
{
  // std::cout << "Finding shortest rows" << std::endl;
  std::map<i_t, std::pair<i_t, i_t>> shortest_rows;
  for (auto col : candidates) {
    auto col_offset    = host_problem.reverse_offsets[col];
    auto nnz_in_col    = host_problem.reverse_offsets[col + 1] - col_offset;
    shortest_rows[col] = {std::numeric_limits<i_t>::max(), -1};
    for (int j = 0; j < nnz_in_col; ++j) {
      auto row      = host_problem.reverse_constraints[col_offset + j];
      auto row_size = host_problem.offsets[row + 1] - host_problem.offsets[row];
      // std::cout << "constraint: " << row << ", lower: " << original_lower_bounds[row]
      //           << ", upper: " << original_upper_bounds[row] << std::endl;
      auto is_inequality =
        host_problem.original_constraint_lower_bounds[row] ==
          -std::numeric_limits<f_t>::infinity() &&
        host_problem.original_constraint_upper_bounds[row] != std::numeric_limits<f_t>::infinity();
      auto is_ranged_or_equality =
        host_problem.original_constraint_lower_bounds[row] !=
          -std::numeric_limits<f_t>::infinity() &&
        host_problem.original_constraint_upper_bounds[row] != std::numeric_limits<f_t>::infinity();
      if (is_inequality || is_ranged_or_equality) {
        auto [min_row_size, row_id] = shortest_rows[col];
        if (row_size < min_row_size) { shortest_rows[col] = std::make_pair(row_size, row); }
      }
    }
  }
  // std::cout << "Shortest rows found" << std::endl;
  return shortest_rows;
}

template <typename i_t, typename f_t>
bool dominated_columns_t<i_t, f_t>::dominates(
  typename problem_t<i_t, f_t>::host_view_t& host_problem, i_t xj, i_t xk, domination_order_t order)
{
  // Signature is valid if any bit set in xj is also set in xk
  // std::cout << "Signature " << xj << " is " << signatures[xj] << std::endl;
  // std::cout << "Signature " << xk << " is " << signatures[xk] << std::endl;
  // std::cout << "Signature " << xj << " and " << xk << " is " << (signatures[xj] & signatures[xk])
  //           << std::endl;
  if ((signatures[xj] & signatures[xk]) != signatures[xj]) { return false; }
  // std::cout << "Signature " << xj << " and " << xk << " is true" << std::endl;

  // Check variable types (iii)
  bool xj_is_int = host_problem.variable_types[xj] == var_t::INTEGER;
  bool xk_is_int = host_problem.variable_types[xk] == var_t::INTEGER;
  if (xj_is_int && !xk_is_int) { return false; }
  // std::cout << "Variable types " << xj << " and " << xk << " is true" << std::endl;

  auto cj = host_problem.objective_coefficients[xj];
  if (order == domination_order_t::NEGATED_XJ) { cj = -cj; }
  auto ck = host_problem.objective_coefficients[xk];
  if (order == domination_order_t::NEGATED_XK) { ck = -ck; }
  // Check objective coefficients (i)
  if (cj > ck) { return false; }
  // std::cout << "Objective coefficients " << xj << " and " << xk << " is true" << std::endl;
  // std::cout << "cj: " << cj << ", ck: " << ck << std::endl;

  // Check constraint coefficients (ii)
  auto xj_offset = host_problem.reverse_offsets[xj];
  auto xj_nnz    = host_problem.reverse_offsets[xj + 1] - xj_offset;
  auto xk_offset = host_problem.reverse_offsets[xk];
  auto xk_nnz    = host_problem.reverse_offsets[xk + 1] - xk_offset;
  // host_problem.print();
  // host_problem.print_transposed();

  for (int i = 0; i < xj_nnz; ++i) {
    auto found_in_row = false;
    auto row1         = host_problem.reverse_constraints[xj_offset + i];
    f_t coeff1        = host_problem.reverse_coefficients[xj_offset + i];
    // std::cout << "cst: " << row1 << ", coeff: " << coeff1 << std::endl;
    f_t coeff2 = 0;

    for (int j = 0; j < xk_nnz; ++j) {
      auto row2 = host_problem.reverse_constraints[xk_offset + j];
      // std::cout << "cst: " << row2
      //           << ", coeff: " << host_problem.reverse_coefficients[xk_offset + j] << std::endl;
      if (row1 == row2) {
        coeff2 = host_problem.reverse_coefficients[xk_offset + j];
        // std::cout << xk << " found in row " << row2 << " with coefficient " << coeff2 <<
        // std::endl;
        found_in_row = true;
        break;
      }
    }
    if (order == domination_order_t::NEGATED_XJ) { coeff1 = -coeff1; }
    if (order == domination_order_t::NEGATED_XK) { coeff2 = -coeff2; }

    // Check the original row bounds to determine constraint type
    f_t row_lb = host_problem.original_constraint_lower_bounds[row1];
    f_t row_ub = host_problem.original_constraint_upper_bounds[row1];

    // Check if this is an equality constraint
    bool is_ranged_or_equality = row_lb != -std::numeric_limits<f_t>::infinity() &&
                                 row_ub != std::numeric_limits<f_t>::infinity();
    // std::cout << "row_lb: " << row_lb << ", row_ub: " << row_ub << std::endl;
    // std::cout << "is_ranged_or_equality: " << is_ranged_or_equality << std::endl;
    // std::cout << "row: " << row1 << ", coeff1: " << coeff1 << ", coeff2: " << coeff2 <<
    // std::endl;

    if (found_in_row && is_ranged_or_equality) {
      // For equality constraints, coefficients must be equal (within epsilon)
      if (std::abs(coeff1 - coeff2) > COEFF_EPSILON) { return false; }
    } else {
      // For inequality constraints, compare coefficients as before
      if (coeff1 > coeff2) { return false; }
    }
  }

  // std::cout << "Constraint coefficients " << xj << " and " << xk << " is true" << std::endl;

  return true;
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::update_variable_bounds(
  typename problem_t<i_t, f_t>::host_view_t& host_problem,
  std::vector<f_t> const& lb_bars,
  std::vector<f_t> const& ub_bars,
  std::vector<i_t> const& h_variable_mapping,
  std::vector<f_t>& h_fixed_var_assignment,
  i_t xj,
  i_t xk,
  domination_order_t order)
{
  // We replaced the strenghtened bounds with inf to apply the theorem. So retrieve the original
  // bounds to apply the lemma and fix variables.
  f_t lj = host_problem.variable_lower_bounds[xj];
  f_t uj = host_problem.variable_upper_bounds[xj];
  f_t lk = host_problem.variable_lower_bounds[xk];
  f_t uk = host_problem.variable_upper_bounds[xk];

  f_t lj_bar = lb_bars[xj];
  f_t uj_bar = ub_bars[xj];
  f_t lk_bar = lb_bars[xk];
  f_t uk_bar = ub_bars[xk];

  auto xj_is_implied_free = lj_bar >= lj && uj_bar <= uj;
  auto xk_is_implied_free = lk_bar >= lk && uk_bar <= uk;

  if (order == domination_order_t::REGULAR) {
    if (uj == std::numeric_limits<f_t>::infinity() || xj_is_implied_free) {
      // case i: xk can be set to lk
      h_fixed_var_assignment[h_variable_mapping[xk]] = lk;
      std::cout << "Fixing variable " << xk << " to lower bound value: " << lk << std::endl;
    } else if (lk == -std::numeric_limits<f_t>::infinity() || xk_is_implied_free) {
      // case iii: xj can be set to uk
      h_fixed_var_assignment[h_variable_mapping[xj]] = uk;
      std::cout << "Fixing variable " << xj << " to upper bound value: " << uk << std::endl;
    }
  } else if (order == domination_order_t::NEGATED_XK) {
    if (uj == std::numeric_limits<f_t>::infinity() || xj_is_implied_free) {
      // case ii: xk can be set to uk
      h_fixed_var_assignment[h_variable_mapping[xk]] = uk;
      std::cout << "Fixing variable " << xk << " to upper bound value: " << uk << std::endl;
    }
  } else if (order == domination_order_t::NEGATED_XJ) {
    if (lk == -std::numeric_limits<f_t>::infinity() || xk_is_implied_free) {
      // case iv: xj can be set to lj
      h_fixed_var_assignment[h_variable_mapping[xj]] = lj;
      std::cout << "Fixing variable " << xj << " to lower bound value: " << lj << std::endl;
    }
  }
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::presolve(bound_presolve_t<i_t, f_t>& bounds_presolve)
{
  auto host_problem = problem.to_host();
  auto lb_bars      = cuopt::host_copy(bounds_presolve.upd.lb, stream);
  auto ub_bars      = cuopt::host_copy(bounds_presolve.upd.ub, stream);

  auto candidates = identify_candidate_variables(host_problem, bounds_presolve, lb_bars, ub_bars);
  // cuopt::print("candidates", candidates);
  std::cout << "candidates size: " << candidates.size() << std::endl;
  if (candidates.empty()) { return; }
  compute_signatures(host_problem);
  auto shortest_rows      = find_shortest_rows(host_problem, candidates);
  auto variable_names     = problem.original_problem_ptr->get_variable_names();
  auto h_variable_mapping = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  // for (const auto& [xj, pair] : shortest_rows) {
  //   std::cout << "processing col " << xj << " (" << variable_names[h_variable_mapping[xj]] << ")"
  //             << std::endl;
  // }
  // std::cout << "shortest_rows size: " << shortest_rows.size() << std::endl;

  // Track variables that have been fixed by domination
  std::vector<i_t> dominated_vars(problem.n_variables, 0);
  auto h_fixed_var_assignment =
    cuopt::host_copy(problem.presolve_data.fixed_var_assignment, stream);

  auto num_dominated_vars = 0;
  for (const auto& [xj, pair] : shortest_rows) {
    if (dominated_vars[xj] == 1) { continue; }
    auto const& [row_size, row] = pair;
    auto row_offset             = host_problem.offsets[row];
    auto nnz_in_row             = host_problem.offsets[row + 1] - row_offset;
    // All the variables in this row are the candidates to be dominated by xj
    for (int j = 0; j < nnz_in_row; ++j) {
      auto xk = host_problem.variables[row_offset + j];
      if (xj == xk || dominated_vars[xk] == 1) { continue; }
      if (dominates(host_problem, xj, xk, domination_order_t::REGULAR)) {
        // Print domination relationship with variable names
        auto xj_name = variable_names[xj];
        auto xk_name = variable_names[xk];
        std::cout << "Domination " << xj << "(" << xj_name << ") -> " << xk << "(" << xk_name << ")"
                  << std::endl;
        update_variable_bounds(host_problem,
                               lb_bars,
                               ub_bars,
                               h_variable_mapping,
                               h_fixed_var_assignment,
                               xj,
                               xk,
                               domination_order_t::REGULAR);
        dominated_vars[xk] = 1;
        ++num_dominated_vars;
      }
    }
  }
  std::cout << "Number of dominated variables: " << num_dominated_vars << std::endl;
  exit(1);

  raft::copy(problem.presolve_data.fixed_var_assignment.data(),
             h_fixed_var_assignment.data(),
             h_fixed_var_assignment.size(),
             stream);
  apply_presolve(problem, presolve_type_t::DOMINATED_COLUMNS, dominated_vars);
}

}  // namespace cuopt::linear_programming::detail

template struct cuopt::linear_programming::detail::dominated_columns_t<int, double>;
