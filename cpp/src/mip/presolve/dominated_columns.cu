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
  : problem(problem_),
    stream(problem.handle_ptr->get_stream()),
    h_implied_lb(problem.n_variables, false),
    h_implied_ub(problem.n_variables, false)
{
}

template <typename i_t, typename f_t>
bool dominated_columns_t<i_t, f_t>::is_ranged_or_equality(f_t lb, f_t ub)
{
  return lb != -std::numeric_limits<f_t>::infinity() && ub != std::numeric_limits<f_t>::infinity();
}

template <typename i_t, typename f_t>
bool dominated_columns_t<i_t, f_t>::is_ge_inequality(f_t lb, f_t ub)
{
  return lb != -std::numeric_limits<f_t>::infinity() && ub == std::numeric_limits<f_t>::infinity();
}

template <typename i_t, typename f_t>
bool dominated_columns_t<i_t, f_t>::is_le_inequality(f_t lb, f_t ub)
{
  return lb == -std::numeric_limits<f_t>::infinity() && ub != std::numeric_limits<f_t>::infinity();
}

template <typename i_t, typename f_t>
std::vector<i_t> dominated_columns_t<i_t, f_t>::identify_candidate_variables(
  typename problem_t<i_t, f_t>::host_view_t& host_problem,
  bound_presolve_t<i_t, f_t>& bounds_presolve)
{
  auto lb = cuopt::host_copy(problem.variable_lower_bounds, stream);
  auto ub = cuopt::host_copy(problem.variable_upper_bounds, stream);
  std::vector<i_t> candidates;
  auto changed_variables = cuopt::host_copy(bounds_presolve.upd.changed_variables, stream);
  auto h_implied_lb      = cuopt::host_copy(bounds_presolve.upd.implied_lb, stream);
  auto h_implied_ub      = cuopt::host_copy(bounds_presolve.upd.implied_ub, stream);
  auto variable_names    = problem.original_problem_ptr->get_variable_names();
  auto variable_mapping  = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  for (int i = 0; i < problem.n_variables; ++i) {
    f_t original_lb = host_problem.original_variable_lower_bounds[variable_mapping[i]];
    f_t original_ub = host_problem.original_variable_upper_bounds[variable_mapping[i]];
    // One of the bounds is infinite we can apply theorem 1.
    if (lb[i] == -std::numeric_limits<f_t>::infinity() ||
        ub[i] == std::numeric_limits<f_t>::infinity()) {
      candidates.push_back(i);
      continue;
    }
    // Check if the bound is implied by the constraints.
    f_t implied_lb = h_implied_lb[i];
    f_t implied_ub = h_implied_ub[i];

    auto is_lower_implied = h_implied_lb[i] != -std::numeric_limits<f_t>::infinity() &&
                            implied_lb - original_lb >= -problem.tolerances.absolute_tolerance;
    auto is_upper_implied = h_implied_ub[i] != std::numeric_limits<f_t>::infinity() &&
                            implied_ub - original_ub <= problem.tolerances.absolute_tolerance;
    h_implied_lb[i] = is_lower_implied;
    h_implied_ub[i] = is_upper_implied;

    if (is_lower_implied || is_upper_implied) { candidates.push_back(i); }
  }
  return candidates;
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::compute_signatures(
  typename problem_t<i_t, f_t>::host_view_t& host_problem)
{
  signatures.resize(problem.n_variables);
  for (int i = 0; i < problem.n_constraints; ++i) {
    auto row_offset = host_problem.offsets[i];
    auto nnz_in_row = host_problem.offsets[i + 1] - row_offset;
    for (int j = 0; j < nnz_in_row; ++j) {
      auto col       = host_problem.variables[row_offset + j];
      auto col_coeff = host_problem.coefficients[row_offset + j];
      if (col_coeff > 0) {
        signatures[col].first.set(i % signature_size);
      } else {
        signatures[col].second.set(i % signature_size);
      }
    }
  }
}

template <typename i_t, typename f_t>
std::map<i_t, std::pair<i_t, i_t>> dominated_columns_t<i_t, f_t>::find_shortest_rows(
  typename problem_t<i_t, f_t>::host_view_t& host_problem, std::vector<i_t> const& candidates)
{
  std::map<i_t, std::pair<i_t, i_t>> shortest_rows;
  for (auto col : candidates) {
    auto col_offset    = host_problem.reverse_offsets[col];
    auto nnz_in_col    = host_problem.reverse_offsets[col + 1] - col_offset;
    shortest_rows[col] = {std::numeric_limits<i_t>::max(), -1};
    for (int j = 0; j < nnz_in_col; ++j) {
      auto row      = host_problem.reverse_constraints[col_offset + j];
      auto row_size = host_problem.offsets[row + 1] - host_problem.offsets[row];
      auto is_inequality_cstr =
        is_ge_inequality(host_problem.original_constraint_lower_bounds[row],
                         host_problem.original_constraint_upper_bounds[row]) ||
        is_le_inequality(host_problem.original_constraint_lower_bounds[row],
                         host_problem.original_constraint_upper_bounds[row]);
      auto is_ranged_or_equality_cstr =
        is_ranged_or_equality(host_problem.original_constraint_lower_bounds[row],
                              host_problem.original_constraint_upper_bounds[row]);
      if (is_inequality_cstr || is_ranged_or_equality_cstr) {
        auto [min_row_size, row_id] = shortest_rows[col];
        if (row_size < min_row_size) { shortest_rows[col] = std::make_pair(row_size, row); }
      }
    }
  }
  return shortest_rows;
}

template <typename i_t, typename f_t>
bool dominated_columns_t<i_t, f_t>::dominates(
  typename problem_t<i_t, f_t>::host_view_t& host_problem,
  i_t xj,
  i_t xk,
  domination_order_t xj_order,
  domination_order_t xk_order)
{
  // Signature is valid if any bit set in xj is also set in xk
  // std::cout << "Signature " << xj << " is " << signatures[xj] << std::endl;
  // std::cout << "Signature " << xk << " is " << signatures[xk] << std::endl;
  // std::cout << "Signature " << xj << " and " << xk << " is " << (signatures[xj] & signatures[xk])
  //           << std::endl;
  auto sj_minus = signatures[xj].first;
  auto sj_plus  = signatures[xj].second;
  auto sk_minus = signatures[xk].first;
  auto sk_plus  = signatures[xk].second;
  if (xj_order == domination_order_t::NEGATED_XJ) { std::swap(sj_minus, sj_plus); }
  if (xk_order == domination_order_t::NEGATED_XK) { std::swap(sk_minus, sk_plus); }
  if ((~sj_minus & sk_minus) != 0) { return false; }
  if ((sj_plus & ~sk_plus) != 0) { return false; }

  // std::cout << "Signature " << xj << " and " << xk << " is true" << std::endl;

  // Check variable types (iii)
  bool xj_is_int = host_problem.variable_types[xj] == var_t::INTEGER;
  bool xk_is_int = host_problem.variable_types[xk] == var_t::INTEGER;
  if (xj_is_int && !xk_is_int) { return false; }
  // std::cout << "Variable types " << xj << " and " << xk << " is true" << std::endl;

  auto cj = host_problem.objective_coefficients[xj];
  if (xj_order == domination_order_t::NEGATED_XJ) { cj = -cj; }
  auto ck = host_problem.objective_coefficients[xk];
  if (xk_order == domination_order_t::NEGATED_XK) { ck = -ck; }
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
    if (xj_order == domination_order_t::NEGATED_XJ) { coeff1 = -coeff1; }
    if (xk_order == domination_order_t::NEGATED_XK) { coeff2 = -coeff2; }

    // Check the original row bounds to determine constraint type
    f_t row_lb = host_problem.original_constraint_lower_bounds[row1];
    f_t row_ub = host_problem.original_constraint_upper_bounds[row1];

    // Check if this is an equality constraint
    bool is_ranged_or_equality_cstr = is_ranged_or_equality(row_lb, row_ub);
    // std::cout << "row_lb: " << row_lb << ", row_ub: " << row_ub << std::endl;
    // std::cout << "is_ranged_or_equality: " << is_ranged_or_equality << std::endl;
    // std::cout << "row: " << row1 << ", coeff1: " << coeff1 << ", coeff2: " << coeff2 <<
    // std::endl;

    if (found_in_row && is_ranged_or_equality_cstr) {
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
  std::vector<i_t> const& h_variable_mapping,
  std::vector<f_t>& h_fixed_var_assignment,
  i_t xj,
  i_t xk,
  domination_order_t xj_order,
  domination_order_t xk_order)
{
  auto variable_mapping = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  std::cout << "original var size: " << host_problem.original_variable_lower_bounds.size()
            << std::endl;
  f_t lj              = host_problem.original_variable_lower_bounds[variable_mapping[xj]];
  f_t uj              = host_problem.original_variable_upper_bounds[variable_mapping[xj]];
  f_t lk              = host_problem.original_variable_lower_bounds[variable_mapping[xk]];
  f_t uk              = host_problem.original_variable_upper_bounds[variable_mapping[xk]];
  auto variable_names = problem.original_problem_ptr->get_variable_names();

  auto xj_is_ub_implied = h_implied_ub[xj];
  auto xj_is_lb_implied = h_implied_lb[xj];
  std::cout << "uj: " << uj << ", lj: " << lj << std::endl;
  std::cout << "xj_is_ub_implied: " << xj_is_ub_implied
            << ", xj_is_lb_implied: " << xj_is_lb_implied << std::endl;
  auto var_fixed = false;

  if (xj_order == domination_order_t::REGULAR && xk_order == domination_order_t::REGULAR) {
    if (uj == std::numeric_limits<f_t>::infinity() || xj_is_ub_implied) {
      // case i: xk can be set to lk
      h_fixed_var_assignment[h_variable_mapping[xk]] = lk;
      cuopt_func_call(var_fixed = true);
      std::cout << "Fixing variable " << variable_mapping[xk] << "(" << variable_names[xk]
                << ") to lower bound value: " << lk << std::endl;
    }
  } else if (xj_order == domination_order_t::REGULAR &&
             xk_order == domination_order_t::NEGATED_XK) {
    if (uj == std::numeric_limits<f_t>::infinity() || xj_is_ub_implied) {
      // case ii: xk can be set to uk
      h_fixed_var_assignment[h_variable_mapping[xk]] = uk;
      cuopt_func_call(var_fixed = true);
      std::cout << "Fixing variable " << variable_mapping[xk] << "(" << variable_names[xk]
                << ") to upper bound value: " << uk << std::endl;
    }
  } else if (xj_order == domination_order_t::NEGATED_XJ &&
             xk_order == domination_order_t::REGULAR) {
    if (lj == -std::numeric_limits<f_t>::infinity() || xj_is_lb_implied) {
      // case iii: xj can be set to lj
      h_fixed_var_assignment[h_variable_mapping[xj]] = lj;
      cuopt_func_call(var_fixed = true);
      std::cout << "Fixing variable " << variable_mapping[xj] << "(" << variable_names[xj]
                << ") to lower bound value: " << lj << std::endl;
    }
  } else if (xj_order == domination_order_t::NEGATED_XJ &&
             xk_order == domination_order_t::NEGATED_XK) {
    if (lj == -std::numeric_limits<f_t>::infinity() || xj_is_lb_implied) {
      // case iv: xj can be set to lj
      h_fixed_var_assignment[h_variable_mapping[xj]] = uk;
      cuopt_func_call(var_fixed = true);
      std::cout << "Fixing variable " << variable_mapping[xj] << "(" << variable_names[xj]
                << ") to upper bound value: " << uk << std::endl;
    }
  } else {
    // std::cout << "xj_order: " << static_cast<int>(xj_order)
    //           << ", xk_order: " << static_cast<int>(xk_order) << ", lj: " << lj << ", uj: " << uj
    //           << ", lk: " << lk << ", uk: " << uk << ", xj_is_lb_implied: " << xj_is_lb_implied
    //           << ", xj_is_ub_implied: " << xj_is_ub_implied
    //           << ", xk_is_lb_implied: " << xk_is_lb_implied
    //           << ", xk_is_ub_implied: " << xk_is_ub_implied << std::endl;
    cuopt_assert(false, "Domination is not possible");
  }
  cuopt_assert(var_fixed, "Variable is not fixed");
}

template <typename i_t, typename f_t>
void dominated_columns_t<i_t, f_t>::presolve(bound_presolve_t<i_t, f_t>& bounds_presolve)
{
  auto host_problem = problem.to_host();

  auto candidates         = identify_candidate_variables(host_problem, bounds_presolve);
  auto h_variable_mapping = cuopt::host_copy(problem.presolve_data.variable_mapping, stream);
  if (candidates.empty()) { return; }
  std::cout << "Number of candidates: " << candidates.size() << std::endl;

  compute_signatures(host_problem);
  auto shortest_rows  = find_shortest_rows(host_problem, candidates);
  auto variable_names = problem.original_problem_ptr->get_variable_names();

  // Track variables that have been fixed by domination
  std::vector<i_t> dominated_vars(problem.n_variables, 0);
  auto h_fixed_var_assignment =
    cuopt::host_copy(problem.presolve_data.fixed_var_assignment, stream);

  auto num_dominated_vars = 0;
  for (const auto& [xj, pair] : shortest_rows) {
    // if (dominated_vars[xj] == 1) { continue; }
    auto const& [row_size, row] = pair;
    auto row_offset             = host_problem.offsets[row];
    auto nnz_in_row             = host_problem.offsets[row + 1] - row_offset;
    auto is_ge_inequality_cstr =
      is_ge_inequality(host_problem.original_constraint_lower_bounds[row],
                       host_problem.original_constraint_upper_bounds[row]);
    auto xj_order =
      is_ge_inequality_cstr ? domination_order_t::NEGATED_XJ : domination_order_t::REGULAR;

    // All the variables in this row are the candidates to be dominated by xj
    for (int j = 0; j < nnz_in_row; ++j) {
      auto xk = host_problem.variables[row_offset + j];
      if (xj == xk) { continue; }
      // if (dominated_vars[xj] == 1) { continue; }
      for (auto xk_order : {domination_order_t::REGULAR, domination_order_t::NEGATED_XK}) {
        auto xj_name = variable_names[h_variable_mapping[xj]];
        auto xk_name = variable_names[h_variable_mapping[xk]];
        // std::cout << "Checking domination " << h_variable_mapping[xj] << "(" << xj_name << ") ->
        // "
        // << h_variable_mapping[xk] << "(" << xk_name << ")" << std::endl;
        if (dominates(host_problem, xj, xk, xj_order, xk_order)) {
          // Print domination relationship with variable names
          std::cout << "Domination " << h_variable_mapping[xj] << "(" << xj_name << ") -> "
                    << h_variable_mapping[xk] << "(" << xk_name << ")" << std::endl;
          update_variable_bounds(
            host_problem, h_variable_mapping, h_fixed_var_assignment, xj, xk, xj_order, xk_order);
          dominated_vars[xk] = 1;
          ++num_dominated_vars;
        }
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
