/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/user_problem.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t>
struct classic_rens_fixing_result_t {
  i_t num_integer_variables{0};
  i_t num_fixed_variables{0};
  i_t num_changed_bounds{0};
};

/**
 * Build the classic RENS neighborhood around an LP solution.
 *
 * Integer variables that are integral within `integer_tol` are fixed to the rounded value. All
 * other integer variables are restricted to the two neighboring integer values. Variables that
 * were already fixed are not included in the returned denominator.
 */
template <typename i_t, typename f_t>
classic_rens_fixing_result_t<i_t> apply_classic_rens_fixings(
  const std::vector<f_t>& solution,
  const std::vector<simplex::variable_type_t>& var_types,
  f_t integer_tol,
  f_t fixed_tol,
  std::vector<f_t>& lower,
  std::vector<f_t>& upper,
  std::vector<bool>& bounds_changed)
{
  assert(solution.size() == var_types.size());
  assert(lower.size() == var_types.size());
  assert(upper.size() == var_types.size());
  assert(bounds_changed.size() == var_types.size());

  classic_rens_fixing_result_t<i_t> result;

  for (i_t j = 0; j < static_cast<i_t>(var_types.size()); ++j) {
    if (var_types[j] == simplex::variable_type_t::CONTINUOUS) { continue; }
    if (std::abs(lower[j] - upper[j]) <= fixed_tol) { continue; }

    ++result.num_integer_variables;

    const f_t old_lower = lower[j];
    const f_t old_upper = upper[j];
    const f_t value     = std::clamp(solution[j], old_lower, old_upper);
    const f_t rounded   = std::round(value);

    if (std::abs(value - rounded) <= integer_tol) {
      const f_t fixed_value = std::clamp(rounded, old_lower, old_upper);
      lower[j]              = fixed_value;
      upper[j]              = fixed_value;
    } else {
      lower[j] = std::clamp(std::floor(value), old_lower, old_upper);
      upper[j] = std::clamp(std::ceil(value), old_lower, old_upper);
    }

    bounds_changed[j] = lower[j] != old_lower || upper[j] != old_upper;
    result.num_changed_bounds += bounds_changed[j];
    result.num_fixed_variables += std::abs(lower[j] - upper[j]) <= fixed_tol;
  }

  return result;
}

}  // namespace cuopt::mathematical_optimization::mip
