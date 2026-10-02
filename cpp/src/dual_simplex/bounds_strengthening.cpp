/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/bounds_strengthening.hpp>

#include <algorithm>
#include <cmath>

namespace cuopt::mathematical_optimization::simplex {

// One step of the Ogita-Rump-Oishi Dot2 algorithm: adds coeff * value to sum and accumulates the
// rounding errors of the product and of the addition in err, so sum + err is accurate to about
// twice the working precision. The product is formed with fma(coeff, value, 0) rather than coeff *
// value, so GCC's default -ffp-contract=fast cannot fuse it into the addition and break the TwoSum.
template <typename f_t>
static inline void dot2_add(f_t coeff, f_t value, f_t& sum, f_t& err)
{
  const f_t h = std::fma(coeff, value, 0.0);
  const f_t t = sum + h;
  const f_t z = t - sum;
  err += ((sum - (t - z)) + (h - z)) + std::fma(coeff, value, -h);
  sum = t;
}

// Computes the min/max activity of row i over the bounds [lower, upper]. Infinite or huge
// contributions are counted in min_inf/max_inf instead of being added to the sum.
template <typename i_t, typename f_t>
void bounds_strengthening_t<i_t, f_t>::compute_row_activity(i_t i,
                                                            const csr_matrix_t<i_t, f_t>& Arow,
                                                            const std::vector<f_t>& lower,
                                                            const std::vector<f_t>& upper)
{
  i_t begin = Arow.row_start[i];
  i_t end   = Arow.row_start[i + 1];
  nnz_processed += end - begin;

  row_activity_t<i_t, f_t> activity;

  for (i_t k = begin; k < end; ++k) {
    i_t j         = Arow.j[k];
    f_t aij       = Arow.x[k];
    f_t bound_max = aij > 0 ? upper[j] : lower[j];
    f_t bound_min = aij < 0 ? upper[j] : lower[j];
    f_t alpha_max = aij * bound_max;
    f_t alpha_min = aij * bound_min;

    if (std::isfinite(alpha_max) && std::abs(alpha_max) < params.huge_value) {
      dot2_add(aij, bound_max, activity.max, activity.max_err);
    } else {
      ++activity.max_inf;
    }

    if (std::isfinite(alpha_min) && std::abs(alpha_min) < params.huge_value) {
      dot2_add(aij, bound_min, activity.min, activity.min_err);
    } else {
      ++activity.min_inf;
    }
  }

  // The peaks track the largest |activity| since the last full computation and are used by
  // update_activities to detect cancellation.
  activity.max_peak = std::abs(activity.max);
  activity.min_peak = std::abs(activity.min);
  row_activities[i] = activity;
}

// Computes the activities of all rows from scratch.
template <typename i_t, typename f_t>
void bounds_strengthening_t<i_t, f_t>::compute_activities(const csr_matrix_t<i_t, f_t>& Arow,
                                                          const std::vector<f_t>& lower,
                                                          const std::vector<f_t>& upper)
{
  row_activities.resize(Arow.m);
  row_queued.resize(Arow.m, false);
  row_queue.clear_resize(std::max(Arow.m, 1));

  for (i_t i = 0; i < Arow.m; ++i) {
    compute_row_activity(i, Arow, lower, upper);
  }
}

// Incrementally updates the activities of every row in column var after its bounds changed from
// [old_lb, old_ub] to [new_lb, new_ub].
template <typename i_t, typename f_t>
void bounds_strengthening_t<i_t, f_t>::update_activities(i_t var,
                                                         f_t old_lb,
                                                         f_t new_lb,
                                                         f_t old_ub,
                                                         f_t new_ub,
                                                         const lp_problem_t<i_t, f_t>& lp,
                                                         const csr_matrix_t<i_t, f_t>& Arow)
{
  i_t begin = lp.A.col_start[var];
  i_t end   = lp.A.col_start[var + 1];
  nnz_processed += end - begin;

  // A wider range can raise the capacity threshold of the rows, so reset it until the row is
  // propagated again.
  const bool loosened = new_ub - new_lb > old_ub - old_lb;

  for (i_t k = begin; k < end; ++k) {
    i_t i                              = lp.A.i[k];
    row_activity_t<i_t, f_t>& activity = row_activities[i];
    if (loosened) { activity.capacity_threshold = inf; }

    // A row waiting for recomputation is rebuilt from the current bounds anyway.
    if (activity.recompute) { continue; }

    f_t aij           = lp.A.x[k];
    f_t old_bound_max = aij > 0 ? old_ub : old_lb;
    f_t old_bound_min = aij < 0 ? old_ub : old_lb;
    f_t new_bound_max = aij > 0 ? new_ub : new_lb;
    f_t new_bound_min = aij < 0 ? new_ub : new_lb;
    f_t old_alpha_max = aij * old_bound_max;
    f_t old_alpha_min = aij * old_bound_min;
    f_t new_alpha_max = aij * new_bound_max;
    f_t new_alpha_min = aij * new_bound_min;

    // Replace the old contribution to the max activity with the new one, moving it between the
    // finite sum and the infinite count when needed.
    if (old_alpha_max != new_alpha_max) {
      if (std::isfinite(old_alpha_max) && std::abs(old_alpha_max) < params.huge_value) {
        dot2_add(-aij, old_bound_max, activity.max, activity.max_err);
      } else {
        --activity.max_inf;
      }

      if (std::isfinite(new_alpha_max) && std::abs(new_alpha_max) < params.huge_value) {
        dot2_add(aij, new_bound_max, activity.max, activity.max_err);
      } else {
        ++activity.max_inf;
      }

      activity.max_peak = std::max(activity.max_peak, std::abs(activity.max));
    }

    // Same for the min activity.
    if (old_alpha_min != new_alpha_min) {
      if (std::isfinite(old_alpha_min) && std::abs(old_alpha_min) < params.huge_value) {
        dot2_add(-aij, old_bound_min, activity.min, activity.min_err);
      } else {
        --activity.min_inf;
      }

      if (std::isfinite(new_alpha_min) && std::abs(new_alpha_min) < params.huge_value) {
        dot2_add(aij, new_bound_min, activity.min, activity.min_err);
      } else {
        ++activity.min_inf;
      }

      activity.min_peak = std::max(activity.min_peak, std::abs(activity.min));
    }

    // An activity much smaller than its peak has lost precision to cancellation in the
    // incremental sum, so mark the row to be recomputed from scratch.
    if (activity.max_peak > params.recompute_factor * std::abs(activity.max) ||
        activity.min_peak > params.recompute_factor * std::abs(activity.min)) {
      activity.recompute = true;
    }
  }
}

// Queues row i unless it is already queued or neither of its sides can tighten a bound. A row
// awaiting recomputation is always queued.
template <typename i_t, typename f_t>
void bounds_strengthening_t<i_t, f_t>::queue_row(i_t i, const lp_problem_t<i_t, f_t>& lp, f_t tol)
{
  if (row_queued[i]) { return; }

  const row_activity_t<i_t, f_t>& activity = row_activities[i];
  if (!activity.recompute) {
    const f_t max_a = activity.max + activity.max_err;
    const f_t min_a = activity.min + activity.min_err;
    const f_t rhs   = lp.rhs[i];
    const bool propagate_upper =
      (activity.max_inf != 0 || max_a > rhs + tol) &&
      (activity.min_inf == 1 ||
       (activity.min_inf == 0 && rhs - min_a <= activity.capacity_threshold));
    const bool propagate_lower =
      (activity.min_inf != 0 || min_a < rhs - tol) &&
      (activity.max_inf == 1 ||
       (activity.max_inf == 0 && max_a - rhs <= activity.capacity_threshold));
    if (!propagate_upper && !propagate_lower) { return; }
  }

  row_queue.push_back(i);
  row_queued[i] = true;
}

// Recomputes all activities from [lower, upper] and propagates every row, tightening lower and
// upper in place.
template <typename i_t, typename f_t>
bool bounds_strengthening_t<i_t, f_t>::propagate_full(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const lp_problem_t<i_t, f_t>& lp,
  std::vector<f_t>& lower,
  std::vector<f_t>& upper)
{
  compute_activities(Arow, lower, upper);
  for (i_t i = 0; i < lp.A.m; ++i) {
    queue_row(i, lp, settings.primal_tol);
  }
  return run_bound_propagation(Arow, var_types, settings, lp, lower, upper);
}

// Propagates from the rows containing a variable marked in bounds_changed. The activities must
// already match [lower, upper], i.e. update_activities was called for every changed variable.
template <typename i_t, typename f_t>
bool bounds_strengthening_t<i_t, f_t>::propagate(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const lp_problem_t<i_t, f_t>& lp,
  const std::vector<bool>& bounds_changed,
  std::vector<f_t>& lower,
  std::vector<f_t>& upper)
{
  assert(bounds_changed.size() == lp.A.n);
  for (i_t j = 0; j < lp.A.n; ++j) {
    if (!bounds_changed[j]) { continue; }
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    nnz_processed += col_end - col_start;
    for (i_t p = col_start; p < col_end; ++p) {
      queue_row(lp.A.i[p], lp, settings.primal_tol);
    }
  }
  return run_bound_propagation(Arow, var_types, settings, lp, lower, upper);
}

// Propagates from the rows containing var. The activities must already match [lower, upper], i.e.
// update_activities was called for var.
template <typename i_t, typename f_t>
bool bounds_strengthening_t<i_t, f_t>::propagate(
  i_t var,
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const lp_problem_t<i_t, f_t>& lp,
  std::vector<f_t>& lower,
  std::vector<f_t>& upper)
{
  const i_t col_start = lp.A.col_start[var];
  const i_t col_end   = lp.A.col_start[var + 1];
  nnz_processed += col_end - col_start;
  for (i_t p = col_start; p < col_end; ++p) {
    queue_row(lp.A.i[p], lp, settings.primal_tol);
  }
  return run_bound_propagation(Arow, var_types, settings, lp, lower, upper);
}

template <typename i_t, typename f_t>
bool bounds_strengthening_t<i_t, f_t>::run_bound_propagation(
  const csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const lp_problem_t<i_t, f_t>& lp,
  std::vector<f_t>& lower,
  std::vector<f_t>& upper)
{
  bool feasible = true;
  i_t iter      = 0;
  while (feasible && !row_queue.empty()) {
    i_t row         = row_queue.pop_front();
    row_queued[row] = false;
    ++iter;

    if (row_activities[row].recompute) { compute_row_activity(row, Arow, lower, upper); }
    row_activity_t activity = row_activities[row];
    const f_t max_a         = activity.max + activity.max_err;
    const f_t min_a         = activity.min + activity.min_err;
    f_t max_tol = settings.primal_tol * std::max({1.0, std::abs(max_a), std::abs(lp.rhs[row])});
    f_t min_tol = settings.primal_tol * std::max({1.0, std::abs(min_a), std::abs(lp.rhs[row])});

    if ((activity.max_inf == 0 && lp.rhs[row] - max_a > max_tol) ||
        (activity.min_inf == 0 && min_a - lp.rhs[row] > min_tol)) {
      settings.log.debug(
        "Iter:: %d, Infeasible constraint %d, rhs %e, min_a %e (%d inf), max_a %e (%d inf)\n",
        iter,
        row,
        lp.rhs[row],
        min_a,
        activity.min_inf,
        max_a,
        activity.max_inf);
      feasible = false;
      break;
    }

    // Smallest absolute change for which a new bound is accepted.
    const f_t tol             = settings.primal_tol;
    const f_t min_improvement = params.min_improvement_factor * tol;
    const f_t rhs             = lp.rhs[row];

    // A side of the row (a x <= rhs or a x >= rhs) is propagated only if it is not redundant and
    // either a single infinite contribution can be bounded or its slack is within the capacity
    // threshold.
    const bool propagate_upper =
      (activity.max_inf != 0 || max_a > rhs + tol) &&
      (activity.min_inf == 1 ||
       (activity.min_inf == 0 && rhs - min_a <= activity.capacity_threshold));
    const bool propagate_lower =
      (activity.min_inf != 0 || min_a < rhs - tol) &&
      (activity.max_inf == 1 ||
       (activity.max_inf == 0 && max_a - rhs <= activity.capacity_threshold));
    if (!propagate_upper && !propagate_lower) { continue; }

    i_t row_start = Arow.row_start[row];
    i_t row_end   = Arow.row_start[row + 1];
    nnz_processed += row_end - row_start;

    // The capacity threshold is recomputed from the bounds seen during this scan.
    f_t threshold = -tol;

    for (i_t p = row_start; p < row_end; ++p) {
      const i_t j    = Arow.j[p];
      const f_t a_ij = Arow.x[p];

      // A fixed variable cannot be tightened.
      const f_t lb = lower[j];
      const f_t ub = upper[j];
      if (lb == ub) { continue; }

      // A continuous column singleton (e.g., a slack) only has bounds implied by this row, so
      // tightening it cannot reach any other row.
      const bool is_integer = !var_types.empty() && var_types[j] == variable_type_t::INTEGER;
      if (!is_integer && lp.A.col_start[j + 1] - lp.A.col_start[j] == 1) { continue; }

      // Largest slack for which x_j can still receive an accepted bound.
      const f_t range = ub - lb;
      const f_t reduction =
        !std::isfinite(range)
          ? inf
          : (is_integer
               ? range - settings.integer_tol
               : range - std::max(params.min_relative_improvement * range, min_improvement));
      threshold = std::max({threshold, std::abs(a_ij) * reduction, tol});

      // Read the activity again for each variable, since earlier tightenings in this row have
      // already updated it, possibly marking it for recomputation.
      if (row_activities[row].recompute) { compute_row_activity(row, Arow, lower, upper); }
      const row_activity_t<i_t, f_t>& current = row_activities[row];

      // Contribution of x_j to the min/max activity, flagged as infinite the same way as in
      // compute_row_activity.
      const f_t bound_min = a_ij < 0 ? ub : lb;
      const f_t bound_max = a_ij > 0 ? ub : lb;
      const f_t alpha_min = a_ij * bound_min;
      const f_t alpha_max = a_ij * bound_max;
      const bool alpha_min_inf =
        !std::isfinite(alpha_min) || std::abs(alpha_min) >= params.huge_value;
      const bool alpha_max_inf =
        !std::isfinite(alpha_max) || std::abs(alpha_max) >= params.huge_value;

      f_t new_lb = lb;
      f_t new_ub = ub;

      // a_ij x_j <= rhs - (min activity of the other variables): an upper bound if a_ij > 0, a
      // lower bound otherwise. If x_j is the only infinite contributor, the residual is the finite
      // sum. The slack rhs - residual is formed with Dot2, since rhs and the residual often nearly
      // cancel.
      if (propagate_upper && (current.min_inf == 0 || (current.min_inf == 1 && alpha_min_inf))) {
        f_t slack     = rhs;
        f_t slack_err = -current.min_err;
        dot2_add(-1.0, current.min, slack, slack_err);
        if (current.min_inf == 0) { dot2_add(a_ij, bound_min, slack, slack_err); }
        const f_t gamma = (slack + slack_err) / a_ij;
        if (std::abs(gamma) < params.max_derived_bound) {
          if (a_ij > 0) {
            new_ub = std::min(new_ub, gamma);
          } else {
            new_lb = std::max(new_lb, gamma);
          }
        }
      }

      // a_ij x_j >= rhs - (max activity of the other variables): a lower bound if a_ij > 0, an
      // upper bound otherwise.
      if (propagate_lower && (current.max_inf == 0 || (current.max_inf == 1 && alpha_max_inf))) {
        f_t slack     = rhs;
        f_t slack_err = -current.max_err;
        dot2_add(-1.0, current.max, slack, slack_err);
        if (current.max_inf == 0) { dot2_add(a_ij, bound_max, slack, slack_err); }
        const f_t gamma = (slack + slack_err) / a_ij;
        if (std::abs(gamma) < params.max_derived_bound) {
          if (a_ij > 0) {
            new_lb = std::max(new_lb, gamma);
          } else {
            new_ub = std::min(new_ub, gamma);
          }
        }
      }

      // Accept a bound only if it improves enough. Integer bounds are rounded first; continuous
      // ones must shrink the range by min_relative_improvement.
      bool tighten_lb = false;
      bool tighten_ub = false;
      if (is_integer) {
        new_lb     = std::ceil(new_lb - settings.integer_tol);
        new_ub     = std::floor(new_ub + settings.integer_tol);
        tighten_lb = new_lb > lb && new_lb - lb > min_improvement * std::abs(new_lb);
        tighten_ub = new_ub < ub && ub - new_ub > min_improvement * std::abs(new_ub);
      } else {
        const f_t lb_range = ub < inf ? ub - lb : std::max(std::abs(lb), std::abs(new_lb));
        const f_t ub_range = lb > -inf ? ub - lb : std::max(std::abs(ub), std::abs(new_ub));
        tighten_lb         = new_lb - min_improvement > lb &&
                     (lb == -inf || new_lb - lb >= params.min_relative_improvement * lb_range);
        tighten_ub = new_ub + min_improvement < ub &&
                     (ub == inf || ub - new_ub >= params.min_relative_improvement * ub_range);
      }

      // Keep the current value on any side that was not accepted.
      if (!tighten_lb && !tighten_ub) { continue; }
      if (!tighten_lb) { new_lb = lb; }
      if (!tighten_ub) { new_ub = ub; }

      // Bounds crossing by more than the tolerance prove infeasibility. A smaller crossing is
      // snapped onto the bound that was not tightened.
      if (new_lb > new_ub) {
        if (new_lb - new_ub > tol * std::max({1.0, std::abs(new_lb), std::abs(new_ub)})) {
          settings.log.debug(
            "Iter:: %d, Infeasible variable %d after propagating row %d, %e > %e\n",
            iter,
            j,
            row,
            new_lb,
            new_ub);
          feasible = false;
          break;
        }
        if (tighten_ub) {
          new_ub = new_lb;
        } else {
          new_lb = new_ub;
        }
      }

      // Apply the new bounds and update the activities of every row containing x_j.
      lower[j] = new_lb;
      upper[j] = new_ub;
      update_activities(j, lb, new_lb, ub, new_ub, lp, Arow);

      // Queue the rows containing x_j, including this one, so they are propagated with the new
      // bounds.
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];
      nnz_processed += col_end - col_start;
      for (i_t q = col_start; q < col_end; ++q) {
        queue_row(lp.A.i[q], lp, tol);
      }
    }

    // An infeasible scan stops early and only saw part of the row.
    if (feasible) { row_activities[row].capacity_threshold = threshold; }
  }

  // Clear the rows left in the queue when infeasibility stops the propagation early.
  while (!row_queue.empty()) {
    row_queued[row_queue.pop_front()] = false;
  }

  last_nnz_processed = nnz_processed;
  nnz_processed      = 0;
  return feasible;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class bounds_strengthening_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::simplex
