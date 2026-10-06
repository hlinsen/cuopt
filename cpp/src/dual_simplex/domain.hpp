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

enum class bound_change_origin_t {
  BRANCH            = 0,  // Regular variable branching
  BOUND_PROPAGATION = 1,  // Bound propagation
  CUT               = 2,  // Cut
  BOUND_IMPLICATION = 3,  // Derived from the implied bounds (via probing cache)
  CLIQUE            = 4,  // Derived from cliques
  OBJECTIVE         = 5,  // Objective cutoff
  REDUCED_COST      = 6,  // Reduced cost strengthening
  SYMMETRY          = 7
};

template <typename i_t, typename f_t>
struct bound_change_t {
  i_t var;
  f_t old_upper;
  f_t new_upper;
  f_t old_lower;
  f_t new_lower;
  bound_change_origin_t origin;

  void apply(std::vector<f_t>& lower, std::vector<f_t>& upper)
  {
    old_lower  = lower[var];
    lower[var] = new_lower;
    old_upper  = upper[var];
    upper[var] = new_upper;
  }
};

struct domain_params {
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

  // Largest slack for which some variable can still receive an accepted bound.
  f_t capacity_threshold = std::numeric_limits<f_t>::infinity();

  bool recompute = false;
};

// Local domain of a node: the row activities over the LP bounds and the stack of bound changes
// that produced those bounds, in the order they were applied.
template <typename i_t, typename f_t>
class domain_t {
 public:
  domain_t() = default;

  // Records a bound change on the stack.
  void push(const bound_change_t<i_t, f_t>& bound_change) { bound_changes.push_back(bound_change); }

  bool apply(lp_problem_t<i_t, f_t>& lp, bound_change_t<i_t, f_t> bound_change)
  {
    if (bound_change.var < 0) { return false; }

    bound_change.apply(lp.lower, lp.upper);
    if (bound_change.old_lower == bound_change.new_lower &&
        bound_change.old_upper == bound_change.new_upper) {
      return false;
    }

    push(bound_change);
    update_activities(lp, bound_change);
    return true;
  }

  bool apply(
    lp_problem_t<i_t, f_t>& lp, i_t var, f_t new_upper, f_t new_lower, bound_change_origin_t origin)
  {
    return apply(lp,
                 {.var = var, .new_upper = new_upper, .new_lower = new_lower, .origin = origin});
  }

  const bound_change_t<i_t, f_t>& operator[](size_t k) const { return bound_changes[k]; }

  size_t size() const { return bound_changes.size(); }
  bool is_empty() const { return bound_changes.empty(); }
  void clear() { bound_changes.clear(); }

  // Applies every recorded change to [lower, upper], in order, refreshing their old bounds.
  void apply_changes(std::vector<f_t>& lower, std::vector<f_t>& upper)
  {
    for (auto& bound_change : bound_changes) {
      bound_change.apply(lower, upper);
    }
  }

  // Pops the changes back to and including the last branching, restoring their bounds in lp and
  // reverting their activities.
  void backtrack_to_parent(lp_problem_t<i_t, f_t>& lp);

  // Recomputes all activities from the bounds in lp. slacks lists the slack column of each row
  // (in any order), or is empty if the rows have no slacks.
  void compute_activities(const csr_matrix_t<i_t, f_t>& Arow,
                          const lp_problem_t<i_t, f_t>& lp,
                          const std::vector<i_t>& slacks);

  void update_activities(const lp_problem_t<i_t, f_t>& lp,
                         const bound_change_t<i_t, f_t>& bound_change);

  void update_activities_from_stack(const lp_problem_t<i_t, f_t>& lp, i_t start = 0);

  bool propagate_full(const csr_matrix_t<i_t, f_t>& Arow,
                      const std::vector<variable_type_t>& var_types,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      lp_problem_t<i_t, f_t>& lp,
                      const std::vector<i_t>& slacks = {});

  // Recomputes all activities from the bounds in lp and propagates from the rows containing vars.
  bool propagate_from_variables(const csr_matrix_t<i_t, f_t>& Arow,
                                const std::vector<variable_type_t>& var_types,
                                const simplex_solver_settings_t<i_t, f_t>& settings,
                                lp_problem_t<i_t, f_t>& lp,
                                const std::vector<i_t>& vars);

  bool propagate_from_stack(const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<variable_type_t>& var_types,
                            const simplex_solver_settings_t<i_t, f_t>& settings,
                            lp_problem_t<i_t, f_t>& lp,
                            i_t start = 0);

  bool apply_and_propagate(const csr_matrix_t<i_t, f_t>& Arow,
                           const std::vector<variable_type_t>& var_types,
                           const simplex_solver_settings_t<i_t, f_t>& settings,
                           bound_change_t<i_t, f_t> bound_change,
                           lp_problem_t<i_t, f_t>& lp);

  size_t last_nnz_processed{0};

 private:
  domain_params params;

  std::vector<row_activity_t<i_t, f_t>> row_activities;
  std::vector<i_t> row_slack;
  std::vector<uint8_t> row_queued;
  circular_deque_t<i_t> row_queue;

  std::vector<bound_change_t<i_t, f_t>> bound_changes;

  size_t nnz_processed{0};

  void queue_row(i_t i, const lp_problem_t<i_t, f_t>& lp, f_t tol);

  bool run_bound_propagation(const csr_matrix_t<i_t, f_t>& Arow,
                             const std::vector<variable_type_t>& var_types,
                             const simplex_solver_settings_t<i_t, f_t>& settings,
                             lp_problem_t<i_t, f_t>& lp);

  void compute_row_activity(i_t i,
                            const csr_matrix_t<i_t, f_t>& Arow,
                            const std::vector<f_t>& lower,
                            const std::vector<f_t>& upper);
};

template <typename i_t, typename f_t>
bool full_bound_strengthening(const csr_matrix_t<i_t, f_t>& Arow,
                              const std::vector<variable_type_t>& var_types,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              lp_problem_t<i_t, f_t>& lp,
                              const std::vector<i_t>& slacks = {})
{
  domain_t<i_t, f_t> domain;
  return domain.propagate_full(Arow, var_types, settings, lp, slacks);
}

}  // namespace cuopt::mathematical_optimization::simplex
