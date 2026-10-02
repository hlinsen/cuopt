/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <dual_simplex/simplex_solver_settings.hpp>

#include <limits>

namespace cuopt::mathematical_optimization::barrier {

using simplex::simplex_solver_settings_t;

// barrier_iterative_refinement = AutoFixedPoint / AutoGMRES, after ClarabelGPU's automatic mode.
// Each solve starts without refinement; it switches on (and stays on) once the iterate nears
// convergence, or once to retry a step that made insufficient progress. ClarabelGPU's kappa/tau
// tests are dropped: there is no homogeneous embedding here. Its duality gap is the relative
// complementarity residual, and its tolerances are the barrier's relative feasibility / optimality
// / complementarity tolerances.
template <typename i_t, typename f_t>
struct auto_iterative_refinement_t {
  enum class state_t { Off, Watch, On };

  state_t state           = state_t::Off;
  i_t watch_iters         = 0;
  bool recovery_attempted = false;
  // Residuals of the previous iterate
  f_t prev_primal_residual = std::numeric_limits<f_t>::infinity();
  f_t prev_dual_residual   = std::numeric_limits<f_t>::infinity();
  f_t prev_gap             = std::numeric_limits<f_t>::infinity();

  bool on() const { return state == state_t::On; }

  void record(f_t primal_residual, f_t dual_residual, f_t gap)
  {
    prev_primal_residual = primal_residual;
    prev_dual_residual   = dual_residual;
    prev_gap             = gap;
  }

  // Off -> Watch -> On, as ClarabelGPU's update_iterative_refinement_state(). Call once per
  // iterate, before record(). Returns true when refinement switches on.
  bool update(const simplex_solver_settings_t<i_t, f_t>& settings,
              i_t iter,
              f_t primal_residual,
              f_t dual_residual,
              f_t gap,
              f_t step_length)
  {
    if (on()) { return false; }
    const f_t primal_tol = settings.barrier_relative_feasibility_tol;
    const f_t dual_tol   = settings.barrier_relative_optimality_tol;
    const f_t gap_tol    = settings.barrier_relative_complementarity_tol;

    const bool reduced_gap_ok = gap < 1e3 * gap_tol;
    const bool watch_gap_ok   = gap < 1e4 * gap_tol;
    const bool watch_feasible =
      primal_residual < 1e6 * primal_tol && dual_residual < 1e6 * dual_tol;

    if (state == state_t::Off) {
      if (!(watch_gap_ok && watch_feasible)) { return false; }
      state       = state_t::Watch;
      watch_iters = 0;
    }
    watch_iters++;

    const bool final_feasible =
      primal_residual < 1e4 * primal_tol && dual_residual < 1e4 * dual_tol;
    const bool primal_regressed =
      prev_primal_residual < 1e6 * primal_tol && primal_residual > 2 * prev_primal_residual;
    const bool dual_regressed =
      prev_dual_residual < 1e6 * dual_tol && dual_residual > 2 * prev_dual_residual;
    // 5x ClarabelGPU's min_switch_step_length
    const bool small_step            = iter > 0 && step_length < 0.05 && reduced_gap_ok;
    const bool watched_one_iteration = watch_iters > 1 && watch_gap_ok && watch_feasible;

    if ((reduced_gap_ok && final_feasible) || watched_one_iteration ||
        (reduced_gap_ok && (primal_regressed || dual_regressed || small_step))) {
      state = state_t::On;
      return true;
    }
    return false;
  }

  // ClarabelGPU's insufficient-progress test: a step that made a residual worse, either after the
  // gap had converged or by more than 100x. Triggers its one-time retry with refinement on.
  bool insufficient_progress(const simplex_solver_settings_t<i_t, f_t>& settings,
                             i_t iter,
                             f_t primal_residual,
                             f_t dual_residual) const
  {
    if (iter <= 1 ||
        !(primal_residual > prev_primal_residual || dual_residual > prev_dual_residual)) {
      return false;
    }
    const f_t primal_tol = settings.barrier_relative_feasibility_tol;
    const f_t dual_tol   = settings.barrier_relative_optimality_tol;
    return prev_gap < settings.barrier_relative_complementarity_tol ||
           (primal_residual > 100 * primal_tol && primal_residual > 100 * prev_primal_residual) ||
           (dual_residual > 100 * dual_tol && dual_residual > 100 * prev_dual_residual);
  }
};

}  // namespace cuopt::mathematical_optimization::barrier
