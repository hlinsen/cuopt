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

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
dominated_columns_t<i_t, f_t>::dominated_columns_t(problem_t<i_t, f_t>& problem_)
  : problem(problem_), stream(problem.handle_ptr->get_stream())
{
}

template <typename i_t, typename f_t>
std::vector<i_t> dominated_columns_t<i_t, f_t>::identify_candidate_variables(
  typename problem_t<i_t, f_t>::host_view_t& host_problem,
  std::vector<f_t> const& lb_bars,
  std::vector<f_t> const& ub_bars)
{
  // std::cout << "Identifying candidate variables" << std::endl;
  auto lb = cuopt::host_copy(problem.variable_lower_bounds, stream);
  auto ub = cuopt::host_copy(problem.variable_upper_bounds, stream);
  std::vector<i_t> candidates;
  for (int i = 0; i < problem.n_variables; ++i) {
    // if (candidates.size() > 10) { break; }
    f_t lb_bar = lb_bars[i];
    f_t ub_bar = ub_bars[i];
    // std::cout << "Variable " << i << " has bounds " << lb_original << " " << ub_original
    //           << " and strengthened bounds " << lb_bar << " " << ub_bar << std::endl;
    // strenghtened bounds are included in original bounds means free
    // It is equivalent to setting the bounds to infinity
    if (lb_bar >= lb[i] && ub_bar <= ub[i]) {
      // host_problem.variable_lower_bounds[i] = -std::numeric_limits<f_t>::infinity();
      // host_problem.variable_upper_bounds[i] = std::numeric_limits<f_t>::infinity();
      // std::cout << "Implied free variable: " << i << std::endl;
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
std::unordered_map<i_t, std::pair<i_t, i_t>> dominated_columns_t<i_t, f_t>::find_shortest_rows(
  typename problem_t<i_t, f_t>::host_view_t& host_problem, std::vector<i_t> const& candidates)
{
  // std::cout << "Finding shortest rows" << std::endl;
  std::unordered_map<i_t, std::pair<i_t, i_t>> shortest_rows;
  auto original_lower_bounds =
    cuopt::host_copy(problem.original_problem_ptr->get_constraint_lower_bounds(), stream);
  auto original_upper_bounds =
    cuopt::host_copy(problem.original_problem_ptr->get_constraint_upper_bounds(), stream);
  for (auto col : candidates) {
    auto col_offset = host_problem.reverse_offsets[col];
    auto nnz_in_col = host_problem.reverse_offsets[col + 1] - col_offset;
    // std::cout << "col: " << col << " has " << nnz_in_col << " non-zeros" << std::endl;
    shortest_rows[col] = {std::numeric_limits<i_t>::max(), -1};
    for (int j = 0; j < nnz_in_col; ++j) {
      auto row      = host_problem.reverse_constraints[col_offset + j];
      auto row_size = host_problem.offsets[row + 1] - host_problem.offsets[row];
      // std::cout << "constraint: " << row << ", lower: " << original_lower_bounds[row]
      //           << ", upper: " << original_upper_bounds[row] << std::endl;
      // Check for LesserThanOrEqual or Equality
      if ((original_lower_bounds[row] == -std::numeric_limits<f_t>::infinity() &&
           original_upper_bounds[row] != std::numeric_limits<f_t>::infinity())) {
        //||
        // problem.integer_equal(original_lower_bounds[row], original_upper_bounds[row])) {
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
  if (!xj_is_int && !xk_is_int) { return false; }
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
    auto row1  = host_problem.reverse_constraints[xj_offset + i];
    f_t coeff1 = host_problem.reverse_coefficients[xj_offset + i];
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
        break;
      }
    }
    if (order == domination_order_t::NEGATED_XJ) { coeff1 = -coeff1; }
    if (order == domination_order_t::NEGATED_XK) { coeff2 = -coeff2; }
    // std::cout << "constraint: " << row1 << ", coeff1: " << coeff1 << ", coeff2: " << coeff2
    //           << std::endl;
    if (coeff1 > coeff2) { return false; }
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
    } else if (lk == -std::numeric_limits<f_t>::infinity() || xk_is_implied_free) {
      // case iii: xj can be set to uk
      h_fixed_var_assignment[h_variable_mapping[xj]] = uk;
    }
  } else if (order == domination_order_t::NEGATED_XK) {
    if (uj == std::numeric_limits<f_t>::infinity() || xj_is_implied_free) {
      // case ii: xk can be set to uk
      h_fixed_var_assignment[h_variable_mapping[xk]] = uk;
    }
  } else if (order == domination_order_t::NEGATED_XJ) {
    if (lk == -std::numeric_limits<f_t>::infinity() || xk_is_implied_free) {
      // case iv: xj can be set to lj
      h_fixed_var_assignment[h_variable_mapping[xj]] = lj;
    }
  }
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::presolve(bound_presolve_t<i_t, f_t>& bounds_presolve)
{
  auto host_problem = problem.to_host();
  auto lb_bars      = cuopt::host_copy(bounds_presolve.upd.lb, stream);
  auto ub_bars      = cuopt::host_copy(bounds_presolve.upd.ub, stream);
  // host_problem.print();
  // cuopt::print("original_variables", problem.original_problem_ptr->get_variable_names());
  // cuopt::print("original_variable_lower_bounds",
  //              problem.original_problem_ptr->get_variable_lower_bounds());
  // cuopt::print("original_variable_upper_bounds",
  //              problem.original_problem_ptr->get_variable_upper_bounds());

  // cuopt::print("variable_lower_bounds", problem.variable_lower_bounds);
  // cuopt::print("variable_upper_bounds", problem.variable_upper_bounds);
  auto candidates = identify_candidate_variables(host_problem, lb_bars, ub_bars);
  // cuopt::print("candidates", candidates);
  if (candidates.empty()) { return; }
  std::cout << "candidates size: " << candidates.size() << std::endl;
  compute_signatures(host_problem);
  auto shortest_rows = find_shortest_rows(host_problem, candidates);
  // std::cout << "shortest_rows size: " << shortest_rows.size() << std::endl;

  // Track variables that have been fixed by domination
  std::vector<i_t> dominated_vars(problem.n_variables, 0);
  auto h_variable_mapping = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  auto h_fixed_var_assignment =
    cuopt::host_copy(problem.presolve_data.fixed_var_assignment, stream);

  for (const auto& [xj, pair] : shortest_rows) {
    // if (dominated_vars[xj] == 1) { continue; }
    auto const& [row_size, row] = pair;
    // std::cout << "For variable " << xj << " the shortest row is " << row << " with size "
    //           << row_size << std::endl;
    auto row_offset = host_problem.offsets[row];
    auto nnz_in_row = host_problem.offsets[row + 1] - row_offset;
    // All the variables in this row are the candidates to be dominated by xj
    for (int j = 0; j < nnz_in_row; ++j) {
      auto xk = host_problem.variables[row_offset + j];
      if (xj == xk) { continue; }
      // std::cout << "Check if " << xj << " dominates " << xk << std::endl;
      for (int order_idx = 0; order_idx < static_cast<int>(domination_order_t::SIZE); ++order_idx) {
        auto order = static_cast<domination_order_t>(order_idx);
        if (order != domination_order_t::REGULAR) { continue; }
        if (dominates(host_problem, xj, xk, order)) {
          std::cout << xj << " dominates " << xk << std::endl;
          update_variable_bounds(host_problem,
                                 lb_bars,
                                 ub_bars,
                                 h_variable_mapping,
                                 h_fixed_var_assignment,
                                 xj,
                                 xk,
                                 order);
          // exit(1);
          dominated_vars[xk] = 1;
        }
      }
    }
  }

  // if (!dominated_vars.empty()) {
  //   std::cout << "Dominated variables: " << dominated_vars.size() << std::endl;
  //   for (size_t var_idx = 0; var_idx < dominated_vars.size(); ++var_idx) {
  //     if (dominated_vars[var_idx] == 0) { continue; }
  //     auto var_type = host_problem.variable_types[var_idx];

  //     if (var_type == var_t::INTEGER) {
  //       std::cout << "("
  //                 << "integer"
  //                 << ", " << var_idx << ") ";
  //     } else {
  //       std::cout << "("
  //                 << "continuous"
  //                 << ", " << var_idx << ") ";
  //     }
  //   }
  //   std::cout << std::endl;
  // }
  raft::copy(problem.presolve_data.fixed_var_assignment.data(),
             h_fixed_var_assignment.data(),
             h_fixed_var_assignment.size(),
             stream);
  apply_presolve(problem, presolve_type_t::DOMINATED_COLUMNS, dominated_vars);
}

}  // namespace cuopt::linear_programming::detail

template struct cuopt::linear_programming::detail::dominated_columns_t<int, double>;
