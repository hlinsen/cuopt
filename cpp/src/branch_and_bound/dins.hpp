/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>
#include <linear_algebra/sparse_vector.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
struct dins_neighborhood_t {
  std::vector<i_t> soft_variables;
  std::vector<f_t> soft_coefficients;
  f_t soft_rhs{};
  i_t num_hard_fixed{};
  i_t num_rebounded{};
};

template <typename i_t>
struct dins_search_state_t {
  explicit dins_search_state_t(i_t radius) : initial_radius(radius), current_radius(radius) {}

  bool advance(bool improved, bool node_limit_reached, bool has_soft_variables)
  {
    if (improved) {
      current_radius = initial_radius;
      return true;
    }
    if (!node_limit_reached || !has_soft_variables) { return false; }

    current_radius -= i_t{5};
    return current_radius >= i_t{0};
  }

  i_t initial_radius;
  i_t current_radius;
};

template <typename i_t>
struct dins_schedule_t {
  static constexpr i_t node_frequency = i_t{100};

  bool should_launch(bool has_incumbent, i_t nodes_explored) const
  {
    return has_incumbent && (!has_launched || nodes_explored - last_launch_node >= node_frequency);
  }

  void record_launch(i_t nodes_explored)
  {
    has_launched     = true;
    last_launch_node = nodes_explored;
  }

  bool has_launched{false};
  i_t last_launch_node{};
};

/**
 * @brief Construct the DINS neighborhood from an incumbent and node relaxation solution.
 *
 * Implements the neighborhood from Ghosh, "DINS, a MIP Improvement Heuristic", IPCO 2007:
 * integer variables at distance at least 0.5 are rebound toward the node relaxation, close general
 * integers are hard-fixed, stable close binaries are hard-fixed, and the remaining binaries form a
 * local-branching inequality with the requested radius.
 */
template <typename i_t, typename f_t>
dins_neighborhood_t<i_t, f_t> build_dins_neighborhood(
  const std::vector<f_t>& node_solution,
  const std::vector<f_t>& root_solution,
  const std::vector<f_t>& incumbent,
  const std::vector<bool>& changed_incumbent,
  const std::vector<simplex::variable_type_t>& var_types,
  f_t fixed_tol,
  f_t integer_tol,
  i_t radius,
  std::vector<f_t>& lower,
  std::vector<f_t>& upper,
  std::vector<bool>& bounds_changed)
{
  const auto n = var_types.size();
  assert(node_solution.size() == n);
  assert(root_solution.size() == n);
  assert(incumbent.size() == n);
  assert(changed_incumbent.size() == n);
  assert(lower.size() == n);
  assert(upper.size() == n);
  assert(bounds_changed.size() == n);

  dins_neighborhood_t<i_t, f_t> neighborhood;
  neighborhood.soft_rhs = static_cast<f_t>(radius);

  for (i_t j = 0; j < static_cast<i_t>(n); ++j) {
    const auto type = var_types[j];
    if (type == simplex::variable_type_t::CONTINUOUS ||
        std::abs(lower[j] - upper[j]) <= fixed_tol) {
      continue;
    }

    // Production presolve currently represents every discrete variable as INTEGER. Infer binary
    // columns from their integral [0, 1] domain so the binary-specific DINS neighborhood is not
    // lost during conversion.
    const bool is_binary = type == simplex::variable_type_t::BINARY ||
                           (lower[j] >= -integer_tol && upper[j] <= f_t{1} + integer_tol);

    const f_t incumbent_value = std::round(incumbent[j]);
    const f_t distance        = std::abs(incumbent_value - node_solution[j]);

    if (distance + integer_tol >= f_t{0.5}) {
      const f_t old_lower = lower[j];
      const f_t old_upper = upper[j];
      if (incumbent_value >= node_solution[j]) {
        lower[j] =
          std::max(lower[j], std::ceil(2 * node_solution[j] - incumbent_value - integer_tol));
        upper[j] = std::min(upper[j], incumbent_value);
      } else {
        lower[j] = std::max(lower[j], incumbent_value);
        upper[j] =
          std::min(upper[j], std::floor(2 * node_solution[j] - incumbent_value + integer_tol));
      }
      if (lower[j] != old_lower || upper[j] != old_upper) {
        bounds_changed[j] = true;
        ++neighborhood.num_rebounded;
      }
      continue;
    }

    const bool stable_binary = is_binary && !changed_incumbent[j] &&
                               std::abs(incumbent_value - node_solution[j]) <= integer_tol &&
                               std::abs(incumbent_value - root_solution[j]) <= integer_tol;

    if (!is_binary || stable_binary) {
      lower[j]          = incumbent_value;
      upper[j]          = incumbent_value;
      bounds_changed[j] = true;
      ++neighborhood.num_hard_fixed;
      continue;
    }

    // Binary local branching: sum_{x*=0} x + sum_{x*=1} (1-x) <= radius.
    neighborhood.soft_variables.push_back(j);
    if (incumbent_value < f_t{0.5}) {
      neighborhood.soft_coefficients.push_back(f_t{1});
    } else {
      neighborhood.soft_coefficients.push_back(f_t{-1});
      neighborhood.soft_rhs -= f_t{1};
    }
  }

  return neighborhood;
}

template <typename i_t, typename f_t>
void append_local_branching_constraint(simplex::user_problem_t<i_t, f_t>& problem,
                                       const dins_neighborhood_t<i_t, f_t>& neighborhood)
{
  if (neighborhood.soft_variables.empty()) { return; }

  csr_matrix_t<i_t, f_t> row_matrix(problem.num_rows, problem.num_cols, 1);
  [[maybe_unused]] const i_t to_row_status = problem.A.to_compressed_row(row_matrix);
  assert(to_row_status == 0);

  sparse_vector_t<i_t, f_t> row(problem.num_cols,
                                static_cast<i_t>(neighborhood.soft_variables.size()));
  row.i                                    = neighborhood.soft_variables;
  row.x                                    = neighborhood.soft_coefficients;
  [[maybe_unused]] const i_t append_status = row_matrix.append_row(row);
  assert(append_status == 0);

  csc_matrix_t<i_t, f_t> column_matrix(
    problem.num_rows + 1, problem.num_cols, row_matrix.row_start.back());
  [[maybe_unused]] const i_t to_col_status = row_matrix.to_compressed_col(column_matrix);
  assert(to_col_status == 0);

  problem.A = std::move(column_matrix);
  ++problem.num_rows;
  problem.rhs.push_back(neighborhood.soft_rhs);
  problem.row_sense.push_back('L');
}

}  // namespace cuopt::mathematical_optimization::mip
