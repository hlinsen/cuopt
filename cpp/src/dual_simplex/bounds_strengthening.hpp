/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <utilities/circular_deque.hpp>

#include <limits>

namespace cuopt::mathematical_optimization::simplex {

struct bounds_strengthening_params {
  double huge_value               = 1e15;
  double recompute_factor         = 1e6;
  double min_relative_improvement = 0.3;
  double min_improvement_factor   = 1e3;
  // Derived bounds larger than this in magnitude are discarded
  double max_derived_bound = 1e8;
};

// The finite parts of the activities are accumulated with compensated (Dot2) summation: the
// activity is max + max_err (resp. min + min_err), where the error terms collect the rounding of
// every product and addition.
template <typename i_t, typename f_t>
struct row_activity_t {
  f_t max      = 0;
  f_t max_err  = 0;
  f_t max_peak = 0;
  i_t max_inf  = 0;

  f_t min      = 0;
  f_t min_err  = 0;
  f_t min_peak = 0;
  i_t min_inf  = 0;

  // Largest slack for which some variable of the row can still receive an accepted bound.
  // It is refreshed whenever the row is propagated and only
  // overestimated in between, so a row whose slack exceeds it can be skipped safely.
  f_t capacity_threshold = std::numeric_limits<f_t>::infinity();

  // Set when cancellation made the incremental sums unreliable. The row is recomputed from scratch
  // before propagation reads it.
  bool recompute = false;
};

template <typename i_t, typename f_t>
class bounds_strengthening_t {
 public:
  // For pure LP bounds strengthening, var_types should be defaulted (i.e. left empty)
  bounds_strengthening_t() = default;

  void compute_row_activity(i_t i,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<f_t>& lower,
                            const std::vector<f_t>& upper);
  void compute_activities(const csr_matrix_t<i_t, f_t>& Arow,
                          const std::vector<f_t>& lower,
                          const std::vector<f_t>& upper);

  void update_activities(i_t var,
                         f_t old_lb,
                         f_t new_lb,
                         f_t old_ub,
                         f_t new_ub,
                         const lp_problem_t<i_t, f_t>& lp,
                         const csr_matrix_t<i_t, f_t>& Arow);

  bool propagate_full(const csr_matrix_t<i_t, f_t>& Arow,
                      const std::vector<variable_type_t>& var_types,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const lp_problem_t<i_t, f_t>& lp,
                      std::vector<f_t>& lower,
                      std::vector<f_t>& upper);

  bool propagate(const csr_matrix_t<i_t, f_t>& Arow,
                 const std::vector<variable_type_t>& var_types,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 const lp_problem_t<i_t, f_t>& lp,
                 const std::vector<bool>& bounds_changed,
                 std::vector<f_t>& lower,
                 std::vector<f_t>& upper);

  bool propagate(i_t var,
                 const csr_matrix_t<i_t, f_t>& Arow,
                 const std::vector<variable_type_t>& var_types,
                 const simplex_solver_settings_t<i_t, f_t>& settings,
                 const lp_problem_t<i_t, f_t>& lp,
                 std::vector<f_t>& lower,
                 std::vector<f_t>& upper);

  size_t last_nnz_processed{0};

 private:
  bounds_strengthening_params params;

  std::vector<row_activity_t<i_t, f_t>> row_activities;
  std::vector<uint8_t> row_queued;
  circular_deque_t<i_t> row_queue;

  size_t nnz_processed{0};

  void queue_row(i_t i, const lp_problem_t<i_t, f_t>& lp, f_t tol);

  bool run_bound_propagation(const csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<variable_type_t>& var_types,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             const lp_problem_t<i_t, f_t>& lp,
                             std::vector<f_t>& lower,
                             std::vector<f_t>& upper);
};

template <typename i_t, typename f_t>
bool full_bound_strengthening(const csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<variable_type_t>& var_types,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              const lp_problem_t<i_t, f_t>& lp,
                              std::vector<f_t>& lower,
                              std::vector<f_t>& upper)
{
  bounds_strengthening_t<i_t, f_t> strengthening;
  return strengthening.propagate_full(Arow, var_types, settings, lp, lower, upper);
}

}  // namespace cuopt::mathematical_optimization::simplex
