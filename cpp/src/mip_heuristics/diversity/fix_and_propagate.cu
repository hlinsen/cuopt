/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "fix_and_propagate.cuh"
#include "fix_propagate_host.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/problem/problem_helpers.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <pdlp/solve.cuh>
#include <utilities/copy_helpers.hpp>

#include <thrust/fill.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <random>
#include <string>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
fix_and_propagate_t<i_t, f_t>::fix_and_propagate_t(mip_solver_context_t<i_t, f_t>& context_,
                                                   problem_t<i_t, f_t>* problem_ptr_,
                                                   population_t<i_t, f_t>& population_,
                                                   local_search_t<i_t, f_t>& ls_,
                                                   cuopt::timer_t& timer_,
                                                   const rmm::device_uvector<f_t>& lp_primal,
                                                   const rmm::device_uvector<f_t>& lp_dual,
                                                   std::function<bool()> preempted_)
  : context(context_),
    problem_ptr(problem_ptr_),
    population(population_),
    ls(ls_),
    timer(timer_),
    lp_optimal_solution(lp_primal),
    lp_dual_optimal_solution(lp_dual),
    preempted(std::move(preempted_))
{
}

template <typename i_t, typename f_t>
typename fix_and_propagate_t<i_t, f_t>::pattern_eval_t
fix_and_propagate_t<i_t, f_t>::eval_fixed_pattern(const std::vector<f_t>& assign,
                                                  f_t tol,
                                                  f_t tlimit,
                                                  const std::vector<f_t>* warm_primal,
                                                  const std::vector<f_t>* warm_dual,
                                                  bool repair,
                                                  f_t repair_tlimit,
                                                  f_t repair_below,
                                                  f_t max_push_time)
{
  raft::common::nvtx::range fun_scope("eval_fixed_pattern");
  pattern_eval_t res;
  auto* handle_ptr  = problem_ptr->handle_ptr;
  auto stream       = handle_ptr->get_stream();
  const i_t n_cstrs = problem_ptr->n_constraints;
  const f_t abs_tol = problem_ptr->tolerances.absolute_tolerance;
  const double t0   = timer.elapsed_time();
  solution_t<i_t, f_t> sol(*problem_ptr);
  sol.copy_new_assignment(assign);
  auto [fixed_problem, fixed_assignment, variable_map] =
    sol.fix_variables(problem_ptr->integer_indices);
  fixed_problem.check_problem_representation(true);
  if (fixed_problem.n_constraints != n_cstrs) { return res; }
  const bool warm = warm_primal != nullptr && warm_dual != nullptr &&
                    warm_primal->size() == (size_t)fixed_problem.n_variables &&
                    warm_dual->size() == (size_t)n_cstrs;
  {
    pdlp_solver_settings_t<i_t, f_t> lp_settings{};
    lp_settings.set_optimality_tolerance(tol);
    lp_settings.time_limit = std::max<double>(1., tlimit);
    lp_settings.method     = method_t::PDLP;
    if (warm) { lp_settings.presolver = presolver_t::None; }
    std::atomic<int> local_halt{0};
    lp_settings.concurrent_halt = &local_halt;
    lp_settings.inside_mip      = false;
    lp_settings.num_gpus        = context.settings.num_gpus;
    if (warm) {
      raft::copy(fixed_assignment.data(), warm_primal->data(), warm_primal->size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(warm_dual->data(), (i_t)warm_dual->size(), stream);
    }
    auto lp_result =
      solve_lp_with_method<i_t, f_t>(fixed_problem, lp_settings, timer_t(lp_settings.time_limit));
    const auto& lp_info = lp_result.get_additional_termination_information();
    res.iters           = lp_info.number_of_steps_taken;
    res.l2_primal_res   = lp_info.l2_primal_residual;
    res.optimal         = lp_result.get_termination_status() == pdlp_termination_status_t::Optimal;
    if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables ||
        lp_result.get_dual_solution().size() != (size_t)n_cstrs) {
      res.time = timer.elapsed_time() - t0;
      return res;
    }
    res.primal = cuopt::host_copy(lp_result.get_primal_solution(), stream);
    res.dual   = cuopt::host_copy(lp_result.get_dual_solution(), stream);
    raft::copy(fixed_assignment.data(),
               lp_result.get_primal_solution().data(),
               fixed_assignment.size(),
               stream);
  }
  clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
  sol.unfix_variables(fixed_assignment, variable_map);
  // full objective (fixed integers included) of the (slightly infeasible) LP point
  res.lp_obj = sol.get_user_objective();
  res.usable = std::isfinite(res.lp_obj);
  // the repair rarely improves the objective of the LP point, so a point that does not beat the
  // incumbent cannot be published: skip the push and the repair (solver-space objectives)
  if (res.usable && population.is_feasible()) {
    const f_t incumbent = population.best_feasible().get_objective();
    if (sol.get_objective() > incumbent + f_t(1e-4) * std::max<f_t>(1., std::abs(incumbent))) {
      res.cannot_beat_incumbent = true;
      res.time                  = timer.elapsed_time() - t0;
      return res;
    }
  }
  if (repair && res.usable && res.lp_obj < repair_below) {
    // phase B: the PID point's l2 primal residual stalls ~50-100 (repair +2%); warm-started
    // pushes with a frozen large primal weight drive it to ~1e-3 at a nearly unchanged objective
    // (the ranking objective above stays the phase-A one)
    const double t_push = timer.elapsed_time();
    struct push_t {
      double weight;
      double cap;
    };
    const double push_scale = std::clamp<double>(max_push_time / 11., 0., 1.);
    const push_t pushes[]   = {{1e4, 6. * push_scale}, {1e5, 5. * push_scale}};
    auto cur_p              = cuopt::device_copy(res.primal, stream);
    auto cur_d              = cuopt::device_copy(res.dual, stream);
    bool pushed             = false;
    for (const auto& ps : pushes) {
      if (timer.check_time_limit() || check_b_b_preemption() || timer.remaining_time() < 30.) {
        break;
      }
      pdlp_solver_settings_t<i_t, f_t> lp_settings{};
      lp_settings.set_optimality_tolerance(1e-7);
      lp_settings.time_limit = ps.cap;
      lp_settings.method     = method_t::PDLP;
      lp_settings.presolver  = presolver_t::None;
      std::atomic<int> local_halt{0};
      lp_settings.concurrent_halt = &local_halt;
      lp_settings.inside_mip      = false;
      lp_settings.num_gpus        = context.settings.num_gpus;
      raft::copy(fixed_assignment.data(), cur_p.data(), cur_p.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(cur_d.data(), (i_t)cur_d.size(), stream);
      lp_settings.set_initial_primal_weight(ps.weight);
      lp_settings.hyper_params.restart_k_p = 0.;
      lp_settings.hyper_params.restart_k_i = 0.;
      lp_settings.hyper_params.restart_k_d = 0.;
      auto lp_result =
        solve_lp_with_method<i_t, f_t>(fixed_problem, lp_settings, timer_t(lp_settings.time_limit));
      if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables ||
          lp_result.get_dual_solution().size() != (size_t)n_cstrs) {
        break;
      }
      res.pushed_l2_primal_res =
        lp_result.get_additional_termination_information().l2_primal_residual;
      raft::copy(cur_p.data(), lp_result.get_primal_solution().data(), cur_p.size(), stream);
      raft::copy(cur_d.data(), lp_result.get_dual_solution().data(), cur_d.size(), stream);
      pushed = true;
    }
    if (pushed) {
      raft::copy(fixed_assignment.data(), cur_p.data(), cur_p.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      sol.unfix_variables(fixed_assignment, variable_map);
      res.pushed_lp_obj = sol.get_user_objective();
    }
    res.push_time = timer.elapsed_time() - t_push;
    bool feas     = sol.get_feasible();
    if (!feas) {
      timer_t rtimer(repair_tlimit);
      thrust::fill(handle_ptr->get_thrust_policy(),
                   fixed_problem.objective_coefficients.begin(),
                   fixed_problem.objective_coefficients.end(),
                   f_t(0));
      fixed_problem.presolve_data.objective_offset = 0;
      auto& lp_state                               = fixed_problem.lp_state;
      lp_state.resize(fixed_problem, stream);
      thrust::fill(handle_ptr->get_thrust_policy(),
                   lp_state.prev_dual.begin(),
                   lp_state.prev_dual.end(),
                   f_t(0));
      relaxed_lp_settings_t r_settings;
      r_settings.tolerance               = 0.5 * abs_tol;
      r_settings.return_first_feasible   = true;
      r_settings.save_state              = true;
      r_settings.check_infeasibility     = false;
      r_settings.per_constraint_residual = true;
      r_settings.has_initial_primal      = true;
      for (int chunk_id = 0; chunk_id < 3 && !feas; ++chunk_id) {
        if (rtimer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) {
          break;
        }
        r_settings.time_limit =
          std::min<double>(rtimer.remaining_time(), timer.remaining_time() - 1.);
        if (r_settings.time_limit < 1.) { break; }
        get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, r_settings);
        sol.unfix_variables(fixed_assignment, variable_map);
        feas = sol.get_feasible();
      }
    }
    if (feas) {
      res.repaired_obj        = sol.get_user_objective();
      res.repaired_assignment = sol.get_host_assignment();
      // like the other heuristics, hand every feasible solution to the population and let it decide
      res.published = population.add_solution(solution_t<i_t, f_t>(sol)).second;
      if (res.published) {
        CUOPT_LOG_INFO(
          "New solution from primal heuristics (CPU fix-and-propagate). Objective %+.6e. Time %.3f",
          res.repaired_obj,
          timer.elapsed_time());
      }
    }
  }
  res.time = timer.elapsed_time() - t0;
  return res;
}

template <typename i_t, typename f_t>
bool fix_and_propagate_t<i_t, f_t>::run()
{
  raft::common::nvtx::range fun_scope("run_root_fix_and_propagate");
  if (problem_ptr->n_integer_vars == 0 || problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return false;
  }
  const auto& hp       = context.settings.heuristic_params;
  const f_t max_budget = std::min<f_t>(hp.fix_propagate_max_time,
                                       hp.fix_propagate_time_ratio * context.settings.time_limit);
  if (max_budget <= 1.) { return false; }
  CUOPT_LOG_DEBUG(
    "Root fix-and-propagate start: budget=%.1fs elapsed=%.2f", max_budget, timer.elapsed_time());
  auto* handle_ptr = problem_ptr->handle_ptr;
  auto stream      = handle_ptr->get_stream();
  const f_t budget = std::min<f_t>(timer.remaining_time() * 0.8, max_budget);
  timer_t fpc_timer(budget);
  const f_t int_tol    = problem_ptr->tolerances.integrality_tolerance;
  const double t_build = timer.elapsed_time();

  // host copies of the model and the LP optimum
  const i_t n_vars    = problem_ptr->n_variables;
  const i_t n_cstrs   = problem_ptr->n_constraints;
  auto h_offsets      = cuopt::host_copy(problem_ptr->offsets, stream);
  auto h_cols         = cuopt::host_copy(problem_ptr->variables, stream);
  auto h_vals         = cuopt::host_copy(problem_ptr->coefficients, stream);
  auto h_clb          = cuopt::host_copy(problem_ptr->constraint_lower_bounds, stream);
  auto h_cub          = cuopt::host_copy(problem_ptr->constraint_upper_bounds, stream);
  auto h_int          = cuopt::host_copy(problem_ptr->integer_indices, stream);
  auto [h_vlb, h_vub] = cuopt::extract_host_bounds<f_t>(problem_ptr->variable_bounds, handle_ptr);
  // reference point: the root LP optimum
  auto h_lp = cuopt::host_copy(lp_optimal_solution, stream);
  host_fix_propagate_t<i_t, f_t> fp_host;
  fp_host.abs_tol                    = problem_ptr->tolerances.absolute_tolerance;
  fp_host.continuous_min_improvement = hp.fix_propagate_continuous_min_improvement;
  fp_host.build(n_vars, n_cstrs, h_offsets, h_cols, h_vals, h_clb, h_cub, h_vlb, h_vub, h_int);
  fp_host.int_tol = problem_ptr->tolerances.integrality_tolerance;
  {
    simplex::user_problem_t<i_t, f_t> user_problem(problem_ptr->handle_ptr);
    problem_ptr->get_host_user_problem(user_problem);
    fp_host.build_domain(user_problem, problem_ptr->clique_table.get());
  }
  // a pass may process max(3e8, 20 x the model's nonzeros) nonzeros before it gives up: measured
  // passes stayed below 1.2e8 (up to 350 x nnz on small models) while a pathological one with
  // very dense integer rows processed 2.5e9 in 54 s (Arow is built by build_domain)
  constexpr int64_t pass_work_floor   = 300000000;
  constexpr int64_t pass_work_per_nnz = 20;
  const int64_t model_nnz = std::max<int64_t>(1, (int64_t)fp_host.Arow.row_start.back());
  fp_host.work_limit      = std::max(pass_work_floor, pass_work_per_nnz * model_nnz);
  i_t n_frac = 0, n_at_one = 0;
  for (auto v : h_int) {
    f_t x = h_lp[v];
    if ((x - std::floor(x) > int_tol) && (std::ceil(x) - x > int_tol)) n_frac++;
    if (x >= 1. - int_tol) n_at_one++;
  }
  CUOPT_LOG_DEBUG(
    "FPC start: budget=%.1fs n_int=%zu lp_fractional=%d lp_ge_one=%d int_rows=%d "
    "(int_only=%d mixed=%d) int_row_nnz=%zu build=%.2fs elapsed=%.2f",
    budget,
    h_int.size(),
    n_frac,
    n_at_one,
    fp_host.n_rows(),
    fp_host.n_binary_only_rows,
    fp_host.n_mixed_rows,
    fp_host.row_cols.size(),
    timer.elapsed_time() - t_build,
    timer.elapsed_time());

  // LP-guided fix-and-propagate: one pass with the configured variable / value strategies
  // (see fix_propagate_host.hpp), the pattern evaluated by a short cost-aware fixed LP (its
  // objective settles within a few thousand PDLP iterations even when the residual stalls),
  // then pushed, repaired and added to the population. All steps are scaled to the budget.
  auto h_obj  = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
  auto h_dual = cuopt::host_copy(lp_dual_optimal_solution, stream);
  std::vector<f_t> red(n_vars, 0);
  std::vector<i_t> uplocks(n_vars, 0), downlocks(n_vars, 0);
  std::vector<char> is_binary(n_vars, 0);
  for (auto v : h_int) {
    f_t aty = 0;
    for (i_t kk = fp_host.var_offsets[v]; kk < fp_host.var_offsets[v + 1]; ++kk) {
      const i_t r = fp_host.var_rows[kk];
      const f_t a = fp_host.var_vals[kk];
      aty += a * h_dual[fp_host.row_orig[r]];
      if ((a > 0 && std::isfinite(fp_host.row_ub[r])) ||
          (a < 0 && std::isfinite(fp_host.row_lb[r]))) {
        uplocks[v]++;
      }
      if ((a > 0 && std::isfinite(fp_host.row_lb[r])) ||
          (a < 0 && std::isfinite(fp_host.row_ub[r]))) {
        downlocks[v]++;
      }
    }
    red[v]       = h_obj[v] - aty;
    is_binary[v] = h_vlb[v] == 0 && h_vub[v] == 1;
  }
  struct cand_t {
    std::string name;  // log name
    fp_variable_strategy_t var_strategy;
    fp_value_strategy_t value_strategy;
    f_t value_param;  // reduced-cost or fraction threshold of the value strategy
    unsigned seed;    // randomized value strategies (fp_randomized)
  };
  std::vector<std::vector<f_t>> seen;
  f_t best_lp = std::numeric_limits<f_t>::infinity();
  int n_eval  = 0;
  std::string best_name;
  std::vector<f_t> best_repaired;
  constexpr f_t rank_tol = 1e-4;
  const f_t rank_tlimit  = std::min<f_t>(8., 0.25 * budget);
  const f_t push_tlimit  = std::min<f_t>(11., 0.3 * budget);
  // a repair that succeeds takes about a second; one that fails would use the whole cap
  const f_t repair_tlimit = std::min<f_t>(5., 0.4 * budget);
  auto run_candidate      = [&](const cand_t& c, const std::vector<f_t>& x) {
    if (fpc_timer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption() ||
        timer.remaining_time() < 2. * budget) {
      return;
    }
    std::vector<i_t> ord(h_int.begin(), h_int.end());
    std::vector<f_t> key(n_vars, 0);
    std::mt19937 g(c.seed);
    std::uniform_real_distribution<double> U(0., 1.);
    for (auto v : h_int) {
      key[v] = fp_order_key<i_t, f_t>(c.var_strategy, x[v], red[v], uplocks[v], downlocks[v]);
    }
    const bool group = fp_groups_by_type(c.var_strategy);
    std::stable_sort(ord.begin(), ord.end(), [&](i_t a, i_t b) {
      if (group && is_binary[a] != is_binary[b]) { return is_binary[a] > is_binary[b]; }
      return key[a] > key[b];
    });
    std::vector<f_t> pref(n_vars, 0.);
    const bool randomized = fp_randomized(c.value_strategy);
    const bool dynamic    = fp_dynamic_locks(c.value_strategy);
    std::vector<char> dyn;
    std::vector<f_t> dyn_up, dyn_down;
    if (dynamic) {
      dyn.assign(n_vars, 0);
      dyn_up.assign(n_vars, 0);
      dyn_down.assign(n_vars, 0);
    }
    for (auto v : h_int) {
      const f_t u = randomized ? f_t(U(g)) : f_t(0);
      pref[v]     = fp_preferred_value<f_t>(c.value_strategy,
                                        x[v],
                                        h_vlb[v],
                                        h_vub[v],
                                        red[v],
                                        c.value_param,
                                        int_tol,
                                        u,
                                        uplocks[v],
                                        downlocks[v],
                                        h_obj[v]);
      if (dynamic) {
        dyn[v] = fp_dynamic_lock_values<f_t>(
          c.value_strategy, x[v], h_vub[v], int_tol, dyn_up[v], dyn_down[v]);
      }
    }
    const double t_fp = timer.elapsed_time();
    i_t n_flipped = 0, n_implied = 0;
    fp_host.work = 0;
    i_t n_conflicts =
      dynamic ? fp_host.fix_and_propagate(ord, pref, n_flipped, n_implied, &dyn, &dyn_up, &dyn_down)
                   : fp_host.fix_and_propagate(ord, pref, n_flipped, n_implied);
    std::vector<f_t> pat(h_int.size());
    f_t int_cost = 0;
    i_t n_ones   = 0;
    for (size_t q = 0; q < h_int.size(); ++q) {
      pat[q] = fp_host.col_lb[h_int[q]];
      int_cost += h_obj[h_int[q]] * pat[q];
      n_ones += pat[q] > 0.5;
    }
    bool dup = false;
    for (auto& s : seen) {
      if (s == pat) { dup = true; }
    }
    CUOPT_LOG_DEBUG(
      "Fix-and-propagate pass %s: time=%.3fs conflicts=%d flipped=%d implied_off_pref=%d ones=%d "
           "int_cost=%g dup=%d cont_tightenings=%ld work=%ld (%.1fx nnz) work_limit_hit=%d "
           "elapsed=%.2f",
      c.name.c_str(),
      timer.elapsed_time() - t_fp,
      n_conflicts,
      n_flipped,
      n_implied,
      n_ones,
      int_cost,
      (int)dup,
      (long)fp_host.n_continuous_tightenings,
      (long)fp_host.work,
      (double)fp_host.work / (double)model_nnz,
      (int)fp_host.work_limit_hit,
      timer.elapsed_time());
    if (dup) { return; }
    seen.push_back(pat);
    std::vector<f_t> assign = x;
    for (size_t q = 0; q < h_int.size(); ++q) {
      assign[h_int[q]] = pat[q];
    }
    // feasibility jump hand-off of a pattern the pass or its evaluation left infeasible
    auto fj_repair = [&]() {
      if (hp.fix_propagate_repair == 0 || fpc_timer.check_time_limit() ||
          timer.check_time_limit() || check_b_b_preemption()) {
        return;
      }
      auto r = fj_repair_pattern(assign, repair_tlimit);
      CUOPT_LOG_DEBUG("Fix-and-propagate FJ repair %s: feasible=%d obj=%g published=%d time=%.2fs",
                      c.name.c_str(),
                      (int)r.feasible,
                      r.obj,
                      (int)r.published,
                      r.time);
      if (r.feasible && best_repaired.empty()) {
        best_name     = c.name + "+fj";
        best_repaired = std::move(r.assignment);
      }
    };
    if (n_conflicts > 0) {
      fj_repair();
      return;
    }
    auto ev = eval_fixed_pattern(
      assign, rank_tol, rank_tlimit, nullptr, nullptr, true, repair_tlimit, best_lp, push_tlimit);
    n_eval++;
    const bool better = ev.usable && ev.lp_obj < best_lp;
    CUOPT_LOG_DEBUG(
      "Pattern rank %s: ones=%d lp_obj=%g optimal=%d iters=%d l2_primal_res=%g time=%.2fs "
           "better=%d pushed_lp_obj=%g pushed_res=%g push_time=%.2fs repaired=%g published=%d "
           "cannot_beat=%d elapsed=%.2f",
      c.name.c_str(),
      n_ones,
      ev.lp_obj,
      (int)ev.optimal,
      ev.iters,
      ev.l2_primal_res,
      ev.time,
      (int)better,
      ev.pushed_lp_obj,
      ev.pushed_l2_primal_res,
      ev.push_time,
      ev.repaired_obj,
      (int)ev.published,
      (int)ev.cannot_beat_incumbent,
      timer.elapsed_time());
    if (better && !ev.repaired_assignment.empty()) {
      best_lp       = ev.lp_obj;
      best_name     = c.name;
      best_repaired = std::move(ev.repaired_assignment);
    } else if (ev.repaired_assignment.empty() && !ev.cannot_beat_incumbent) {
      fj_repair();
    }
  };
  const cand_t configured{"lp_guided",
                          static_cast<fp_variable_strategy_t>(hp.fix_propagate_variable_strategy),
                          static_cast<fp_value_strategy_t>(hp.fix_propagate_value_strategy),
                          hp.fix_propagate_value_param,
                          12u};
  run_candidate(configured, h_lp);
  if (best_repaired.empty()) {
    CUOPT_LOG_DEBUG("Root fix-and-propagate: no repaired pattern (evaluated %d) elapsed=%.2f",
                    n_eval,
                    timer.elapsed_time());
    return false;
  }
  CUOPT_LOG_DEBUG("Root fix-and-propagate best: %s lp_obj=%g evaluated=%d elapsed=%.2f",
                  best_name.c_str(),
                  best_lp,
                  n_eval,
                  timer.elapsed_time());
  // the pattern was already pushed, repaired and added to the population by its evaluation
  return true;
}

template <typename i_t, typename f_t>
typename fix_and_propagate_t<i_t, f_t>::fj_repair_t
fix_and_propagate_t<i_t, f_t>::fj_repair_pattern(const std::vector<f_t>& assign, f_t time_limit)
{
  raft::common::nvtx::range fun_scope("fj_repair_pattern");
  fj_repair_t res;
  const double t0 = timer.elapsed_time();
  timer_t rtimer(time_limit);
  solution_t<i_t, f_t> sol(*problem_ptr);
  sol.copy_new_assignment(assign);
  // as for the rounded root LP point in local search: complete the continuous part with a short
  // fixed-integer LP, then let feasibility jump move every variable
  if (problem_ptr->n_integer_vars < problem_ptr->n_variables) {
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit = std::min<f_t>(2., 0.4 * time_limit);
    lp_settings.tolerance  = problem_ptr->tolerances.absolute_tolerance;
    run_lp_with_vars_fixed(*problem_ptr, sol, problem_ptr->integer_indices, lp_settings);
  }
  ls.fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  ls.fj.settings.n_of_minimums_for_exit = 20000;
  ls.fj.settings.update_weights         = true;
  ls.fj.settings.feasibility_run        = false;
  ls.do_fj_solve(sol, ls.fj, std::max<f_t>(rtimer.remaining_time(), 0.1), "fix_propagate_repair");
  sol.compute_feasibility();
  res.feasible = sol.get_feasible();
  if (res.feasible) {
    res.obj        = sol.get_user_objective();
    res.assignment = sol.get_host_assignment();
    res.published  = population.add_solution(solution_t<i_t, f_t>(sol)).second;
    if (res.published) {
      CUOPT_LOG_INFO(
        "New solution from primal heuristics (CPU fix-and-propagate, FJ repair). Objective %+.6e. "
        "Time %.3f",
        res.obj,
        timer.elapsed_time());
    }
  }
  res.time = timer.elapsed_time() - t0;
  return res;
}

#if MIP_INSTANTIATE_FLOAT
template class fix_and_propagate_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class fix_and_propagate_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
