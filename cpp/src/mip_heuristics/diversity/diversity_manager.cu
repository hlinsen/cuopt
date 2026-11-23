/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "cuda_profiler_api.h"
#include "diversity_manager.cuh"

#include <mip_heuristics/feasibility_jump/early_cpufj.cuh>
#include <mip_heuristics/lns/thread_budget.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include <mip_heuristics/presolve/block_bve.cuh>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/presolve/probing_cache.cuh>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/problem/problem_helpers.cuh>

#include <cuopt/mathematical_optimization/solve.hpp>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include <pdlp/root_lp_snapshot.cuh>
#include <pdlp/solve.cuh>
#include "fix_propagate_host.hpp"

#include <utilities/copy_helpers.hpp>
#include <utilities/scope_guard.hpp>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>

constexpr bool fj_only_run = false;

namespace cuopt::mathematical_optimization::mip {

// Private diagnostic bridge: keep the non-exported MIP constructor inside this library.
CUOPT_EXPORT optimization_problem_solution_t<int, double> solve_root_snapshot_diagnostic(
  optimization_problem_t<int, double>& op,
  const pdlp_solver_settings_t<int, double>& settings,
  bool& root_unchanged)
{
  problem_t<int, double> root(op);
  const auto original_matrix = cuopt::host_copy(root.coefficients, root.handle_ptr->get_stream());
  const auto original_objective =
    cuopt::host_copy(root.objective_coefficients, root.handle_ptr->get_stream());
  const auto original_rows =
    cuopt::host_copy(root.constraint_lower_bounds, root.handle_ptr->get_stream());
  auto snapshot = make_root_lp_snapshot(root);
  auto result   = solve_lp(snapshot, settings);
  root_unchanged =
    original_matrix == cuopt::host_copy(root.coefficients, root.handle_ptr->get_stream()) &&
    original_objective ==
      cuopt::host_copy(root.objective_coefficients, root.handle_ptr->get_stream()) &&
    original_rows ==
      cuopt::host_copy(root.constraint_lower_bounds, root.handle_ptr->get_stream()) &&
    result.get_primal_solution().size() == root.objective_coefficients.size() &&
    result.get_dual_solution().size() == root.constraint_lower_bounds.size();
  return result;
}

size_t fp_recombiner_config_t::max_n_of_vars_from_other =
  fp_recombiner_config_t::initial_n_of_vars_from_other;
size_t ls_recombiner_config_t::max_n_of_vars_from_other =
  ls_recombiner_config_t::initial_n_of_vars_from_other;
size_t bp_recombiner_config_t::max_n_of_vars_from_other =
  bp_recombiner_config_t::initial_n_of_vars_from_other;
size_t sub_mip_recombiner_config_t::max_n_of_vars_from_other =
  sub_mip_recombiner_config_t::initial_n_of_vars_from_other;

template <typename i_t, typename f_t>
std::vector<recombiner_enum_t> recombiner_t<i_t, f_t>::enabled_recombiners;

template <typename i_t, typename f_t>
diversity_manager_t<i_t, f_t>::diversity_manager_t(mip_solver_context_t<i_t, f_t>& context_)
  : context(context_),
    branch_and_bound_ptr(nullptr),
    problem_ptr(context.problem_ptr),
    population("population",
               context,
               *this,
               diversity_config.max_var_diff,
               context_.settings.heuristic_params.population_size,
               context_.settings.heuristic_params.initial_infeasibility_weight *
                 context.problem_ptr->n_constraints),
    lp_optimal_solution(context.problem_ptr->n_variables,
                        context.problem_ptr->handle_ptr->get_stream()),
    lp_dual_optimal_solution(context.problem_ptr->n_constraints,
                             context.problem_ptr->handle_ptr->get_stream()),
    ls(context, lp_optimal_solution),
    timer(diversity_config.default_time_limit),
    bound_prop_recombiner(context,
                          context.problem_ptr->n_variables,
                          ls.constraint_prop,
                          context.problem_ptr->handle_ptr),
    fp_recombiner(context,
                  context.problem_ptr->n_variables,
                  ls.fj,
                  ls.constraint_prop,
                  ls.line_segment_search,
                  lp_optimal_solution,
                  context.problem_ptr->handle_ptr),
    line_segment_recombiner(context,
                            context.problem_ptr->n_variables,
                            ls.line_segment_search,
                            context.problem_ptr->handle_ptr),
    sub_mip_recombiner(
      context, population, context.problem_ptr->n_variables, context.problem_ptr->handle_ptr),
    rng(derive_seed(context.base_seed, rng_id_t::diversity_manager, 0)),
    stats(context.stats),
    mab_recombiner(0,
                   derive_seed(context.base_seed, rng_id_t::diversity_manager, 1),
                   recombiner_alpha,
                   "recombiner"),
    mab_ls(mab_ls_config_t<i_t, f_t>::n_of_arms,
           derive_seed(context.base_seed, rng_id_t::diversity_manager, 2),
           ls_alpha,
           "ls"),
    ls_hash_map(*context.problem_ptr)
{
  int max_config             = -1;
  int env_config_id          = -1;
  const char* env_max_config = std::getenv("CUOPT_MAX_CONFIG");
  if (env_max_config != nullptr) {
    try {
      max_config = std::stoi(env_max_config);
      CUOPT_LOG_INFO("Using maximum configuration value from environment: %d", max_config);
    } catch (const std::exception& e) {
      CUOPT_LOG_WARN("Failed to parse CUOPT_MAX_CONFIG environment variable: %s", e.what());
    }
  }

  const char* env_config_id_raw = std::getenv("CUOPT_CONFIG_ID");
  if (env_config_id_raw == nullptr) { return; }

  try {
    env_config_id = std::stoi(env_config_id_raw);
  } catch (const std::exception& e) {
    CUOPT_LOG_WARN("Failed to parse CUOPT_CONFIG_ID environment variable: %s", e.what());
    return;
  }

  if (max_config > 0 && env_config_id >= max_config) {
    CUOPT_LOG_WARN(
      "CUOPT_CONFIG_ID=%d is outside [0, %d). Ignoring cut override.", env_config_id, max_config);
    return;
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::consume_staged_simplex_solution(lp_state_t<i_t, f_t>& lp_state)
{
  std::vector<f_t> staged_simplex_solution_local;
  std::vector<f_t> staged_simplex_dual_solution_local;
  f_t staged_simplex_objective_local = std::numeric_limits<f_t>::infinity();
  {
    std::lock_guard<std::mutex> guard(relaxed_solution_mutex);
    cuopt_assert(simplex_solution_exists.load(),
                 "Simplex solution flag set without a staged simplex solution");
    staged_simplex_solution_local      = staged_simplex_solution;
    staged_simplex_dual_solution_local = staged_simplex_dual_solution;
    staged_simplex_objective_local     = staged_simplex_objective;
  }
  solution_t<i_t, f_t> new_sol(*problem_ptr);
  cuopt_assert(new_sol.assignment.size() == staged_simplex_solution_local.size(),
               "Assignment size mismatch");
  cuopt_assert(problem_ptr->n_constraints == staged_simplex_dual_solution_local.size(),
               "Dual assignment size mismatch");
  new_sol.copy_new_assignment(staged_simplex_solution_local);
  new_sol.compute_feasibility();
  cuopt_assert(integer_equal(new_sol.get_user_objective(), staged_simplex_objective_local, 1e-3),
               "Objective mismatch");
  raft::copy(lp_optimal_solution.data(),
             staged_simplex_solution_local.data(),
             staged_simplex_solution_local.size(),
             problem_ptr->handle_ptr->get_stream());
  clamp_within_var_bounds(lp_optimal_solution, problem_ptr, problem_ptr->handle_ptr);
  raft::copy(lp_state.prev_primal.data(),
             lp_optimal_solution.data(),
             lp_optimal_solution.size(),
             problem_ptr->handle_ptr->get_stream());
  problem_ptr->handle_ptr->sync_stream();
  solution_t<i_t, f_t> bounded_lp_sol(*problem_ptr);
  bounded_lp_sol.copy_new_assignment(lp_optimal_solution);
  bounded_lp_sol.handle_ptr->sync_stream();
  auto max_lp_bound_violation = bounded_lp_sol.compute_max_variable_violation();
  cuopt_assert(max_lp_bound_violation == 0.0,
               "LP optimal solution must be within variable bounds after staged copy");
  set_new_user_bound(staged_simplex_objective_local);
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_local_search(solution_t<i_t, f_t>& solution,
                                                     const weight_t<i_t, f_t>& weights,
                                                     timer_t& timer,
                                                     ls_config_t<i_t, f_t>& ls_config)
{
  raft::common::nvtx::range fun_scope("run_local_search");
  i_t ls_mab_option = mab_ls.select_mab_option();
  mab_ls_config_t<i_t, f_t>::get_local_search_and_lm_from_config(ls_mab_option, ls_config);
  ls_hash_map.insert(solution);
  constexpr i_t skip_solutions_threshold = 3;
  if (ls_hash_map.check_skip_solution(solution, skip_solutions_threshold)) { return false; }
  ls.run_local_search(solution, weights, timer, ls_config);
  return true;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::generate_solution(f_t time_limit, bool random_start)
{
  raft::common::nvtx::range fun_scope("generate_solution");
  solution_t<i_t, f_t> sol(*problem_ptr);
  sol.compute_feasibility();
  // if a feasible is found, it is added to the population
  ls.generate_solution(sol, random_start, &population, time_limit);
  population.add_solution(std::move(sol), "generate_solution");
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::fpc_complete_continuous(solution_t<i_t, f_t>& sol,
                                                            cuopt::timer_t& fpc_timer,
                                                            int attempt,
                                                            bool detect_infeasibility)
{
  raft::common::nvtx::range fun_scope("fpc_complete_continuous");
  auto stream          = problem_ptr->handle_ptr->get_stream();
  const double t_start = timer.elapsed_time();
  auto [fixed_problem, fixed_assignment, variable_map] =
    sol.fix_variables(problem_ptr->integer_indices);
  fixed_problem.check_problem_representation(true);
  // objective-free completion: only primal feasibility of the continuous part matters for an
  // incumbent, and the cost vector (range 7e-12..4e7) is what keeps PDLP from reaching 1e-6
  // per-row accuracy on the fixed LP; the objective of the result is evaluated on the original.
  thrust::fill(problem_ptr->handle_ptr->get_thrust_policy(),
               fixed_problem.objective_coefficients.begin(),
               fixed_problem.objective_coefficients.end(),
               f_t(0));
  fixed_problem.presolve_data.objective_offset = 0;
  auto& lp_state                               = fixed_problem.lp_state;
  lp_state.resize(fixed_problem, stream);
  thrust::fill(problem_ptr->handle_ptr->get_thrust_policy(),
               lp_state.prev_dual.begin(),
               lp_state.prev_dual.end(),
               f_t(0));
  const bool dual_warm = false;
  CUOPT_LOG_INFO(
    "FPC attempt %d completion LP: fixed problem vars=%d cstrs=%d nnz=%d dual_warm=%d build=%.2fs "
    "elapsed=%.2f",
    attempt,
    fixed_problem.n_variables,
    fixed_problem.n_constraints,
    fixed_problem.nnz,
    (int)dual_warm,
    timer.elapsed_time() - t_start,
    timer.elapsed_time());
  relaxed_lp_settings_t lp_settings;
  lp_settings.tolerance               = problem_ptr->tolerances.absolute_tolerance;
  lp_settings.return_first_feasible   = true;
  lp_settings.save_state              = true;
  lp_settings.check_infeasibility     = detect_infeasibility;
  lp_settings.per_constraint_residual = true;
  lp_settings.has_initial_primal      = true;
  // time-chunked completion: each chunk restarts PDLP from the previous iterate (warm start)
  int chunk_id = 0;
  while (!fpc_timer.check_time_limit() && !timer.check_time_limit()) {
    if (check_b_b_preemption()) { return false; }
    const double chunk     = chunk_id == 0 ? 20. : (chunk_id == 1 ? 40. : 60.);
    lp_settings.time_limit = std::min(chunk, fpc_timer.remaining_time());
    if (lp_settings.time_limit < 1.) { break; }
    const double t_chunk = timer.elapsed_time();
    auto resp   = get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, lp_settings);
    auto status = resp.get_termination_status();
    const auto& info = resp.get_additional_termination_information();
    sol.unfix_variables(fixed_assignment, variable_map);
    const bool feas = sol.get_feasible();
    CUOPT_LOG_INFO(
      "FPC attempt %d completion chunk %d: status=%s iters=%d l2_primal_res=%g "
      "l2_dual_res=%g gap=%g chunk_time=%.2fs feasible=%d max_cstr_viol=%g excess=%g obj=%g "
      "elapsed=%.2f",
      attempt,
      chunk_id,
      resp.get_termination_status_string().c_str(),
      info.number_of_steps_taken,
      info.l2_primal_residual,
      info.l2_dual_residual,
      info.gap,
      timer.elapsed_time() - t_chunk,
      (int)feas,
      sol.compute_max_constraint_violation(),
      sol.get_total_excess(),
      sol.get_user_objective(),
      timer.elapsed_time());
    chunk_id++;
    if (feas) {
      f_t max_excess = 0;
      if (strict_point_ok(sol.get_host_assignment(), max_excess) &&
          original_point_ok(sol.get_host_assignment())) {
        return true;
      }
      // keep a margin over the external validator: tighten and continue from this point
      lp_settings.tolerance *= 0.1;
      CUOPT_LOG_INFO("FPC attempt %d completion: strict check failed excess=%g, tol -> %g",
                     attempt,
                     max_excess,
                     lp_settings.tolerance);
      continue;
    }
    if (status == pdlp_termination_status_t::PrimalInfeasible ||
        status == pdlp_termination_status_t::DualInfeasible ||
        status == pdlp_termination_status_t::NumericalError) {
      return false;
    }
  }
  return false;
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::polish_continuous(solution_t<i_t, f_t>& in_sol,
                                                      f_t budget,
                                                      flip_lp_state_t* out_state,
                                                      int first_stage,
                                                      const std::vector<f_t>* init_primal,
                                                      const std::vector<f_t>* init_dual,
                                                      int last_stage,
                                                      double stage0_cap)
{
  raft::common::nvtx::range fun_scope("polish_continuous");
  if (!in_sol.get_feasible() || problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return false;
  }
  budget = std::min<f_t>(budget, timer.remaining_time() * 0.8);
  if (budget < 5.) { return false; }
  timer_t ptimer(budget);
  auto* handle_ptr = problem_ptr->handle_ptr;
  auto stream      = handle_ptr->get_stream();
  solution_t<i_t, f_t> sol(in_sol);
  const f_t obj0       = sol.get_user_objective();
  f_t best_obj         = obj0;
  bool improved        = false;
  const double t_start = timer.elapsed_time();
  auto [fixed_problem, fixed_assignment, variable_map] =
    sol.fix_variables(problem_ptr->integer_indices);
  fixed_problem.check_problem_representation(true);
  auto publish_polished = [&](solution_t<i_t, f_t>& s, const char* origin) {
    if (fpc_external_publish) {
      population.add_external_solution(
        s.get_host_assignment(), s.get_objective(), solution_origin_t::POLISH);
    } else {
      population.add_solution(solution_t<i_t, f_t>(s), origin);
    }
  };

  // two-phase cost-aware fixed LP: stage 0 = Stable3 PID at 1e-4 (settles the objective; its
  // primal weight drifts low and the l2 primal residual stalls ~100, so repair costs +2-5%),
  // then warm-started PDLP pushes (no presolve) with a frozen large primal weight that drive
  // the residual to ~1e-3 at nearly unchanged objective, so the repair moves the point little.
  struct stage_t {
    double tol;
    double budget_frac;
    double lp_cap;         // seconds; 0 = none
    double primal_weight;  // 0 = solver default (PID-controlled)
  };
  const stage_t stages[]    = {{1e-4, 0.35, 0., 0.}, {1e-7, 0.6, 15., 1e4}, {1e-7, 1.0, 10., 1e5}};
  constexpr size_t n_stages = 3;
  const size_t k_end = last_stage < 0 ? n_stages : std::min(n_stages, (size_t)last_stage + 1);
  std::vector<optimization_problem_t<i_t, f_t>> snapshots;
  for (size_t k = 0; k < n_stages; ++k) {
    snapshots.emplace_back(make_root_lp_snapshot(fixed_problem));
  }
  // stage k > 0 (or stage 0 given an initial primal/dual) continues PDLP from the previous
  // primal/dual (no presolve, so the iterate maps 1:1 onto the fixed problem)
  rmm::device_uvector<f_t> warm_primal(0, stream);
  rmm::device_uvector<f_t> warm_dual(0, stream);
  if (init_primal != nullptr && init_dual != nullptr &&
      init_primal->size() == (size_t)fixed_problem.n_variables &&
      init_dual->size() == (size_t)fixed_problem.n_constraints) {
    warm_primal = cuopt::device_copy(*init_primal, stream);
    warm_dual   = cuopt::device_copy(*init_dual, stream);
  }
  // objective-free repair problem (fixed_problem itself)
  thrust::fill(handle_ptr->get_thrust_policy(),
               fixed_problem.objective_coefficients.begin(),
               fixed_problem.objective_coefficients.end(),
               f_t(0));
  fixed_problem.presolve_data.objective_offset = 0;

  for (size_t k = (size_t)std::max(0, first_stage); k < k_end; ++k) {
    if (ptimer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) { break; }
    pdlp_solver_settings_t<i_t, f_t> lp_settings{};
    lp_settings.set_optimality_tolerance(stages[k].tol);
    lp_settings.time_limit = std::max<double>(
      1., ptimer.remaining_time() * stages[k].budget_frac * (k + 1 == k_end ? 0.8 : 1.));
    if (k == 0 && stage0_cap > 0.) {
      lp_settings.time_limit = std::min<double>(lp_settings.time_limit, stage0_cap);
    }
    if (stages[k].lp_cap > 0.) {
      lp_settings.time_limit = std::min<double>(lp_settings.time_limit, stages[k].lp_cap);
    }
    lp_settings.method = context.settings.method;
    std::atomic<int> local_halt{0};
    lp_settings.concurrent_halt = &local_halt;
    lp_settings.inside_mip      = false;
    lp_settings.num_gpus        = context.settings.num_gpus;
    const bool warm             = warm_primal.size() == (size_t)fixed_problem.n_variables &&
                      warm_dual.size() == (size_t)fixed_problem.n_constraints;
    if (warm) {
      lp_settings.method    = method_t::PDLP;
      lp_settings.presolver = presolver_t::None;
      // the snapshot carries no variable types; the initial-solution check reads them
      std::vector<var_t> h_types(fixed_problem.n_variables, var_t::CONTINUOUS);
      snapshots[k].set_variable_types(h_types.data(), fixed_problem.n_variables);
      raft::copy(fixed_assignment.data(), warm_primal.data(), warm_primal.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(warm_dual.data(), (i_t)warm_dual.size(), stream);
      if (stages[k].primal_weight > 0.) {
        lp_settings.set_initial_primal_weight(stages[k].primal_weight);
        lp_settings.hyper_params.restart_k_p = 0.;
        lp_settings.hyper_params.restart_k_i = 0.;
        lp_settings.hyper_params.restart_k_d = 0.;
      }
    }
    CUOPT_LOG_INFO(
      "Polish stage %d start: obj=%g tol=%g vars=%d cstrs=%d budget=%.1fs lp_budget=%.1fs "
      "warm=%d primal_weight=%g elapsed=%.2f",
      (int)k,
      best_obj,
      stages[k].tol,
      fixed_problem.n_variables,
      fixed_problem.n_constraints,
      budget,
      lp_settings.time_limit,
      (int)warm,
      warm ? stages[k].primal_weight : 0.,
      timer.elapsed_time());
    const double t_lp   = timer.elapsed_time();
    auto lp_result      = solve_lp(snapshots[k], lp_settings);
    const auto& lp_info = lp_result.get_additional_termination_information();
    CUOPT_LOG_INFO(
      "Polish stage %d LP: status=%s method=%s lp_obj=%g iters=%d l2_primal_res=%g time=%.2fs "
      "elapsed=%.2f",
      (int)k,
      lp_result.get_termination_status_string().c_str(),
      method_to_string(lp_info.solved_by).c_str(),
      (double)lp_result.get_objective_value(),
      lp_info.number_of_steps_taken,
      lp_info.l2_primal_residual,
      timer.elapsed_time() - t_lp,
      timer.elapsed_time());
    if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables) {
      CUOPT_LOG_INFO("Polish stage %d LP returned no usable primal", (int)k);
      continue;
    }
    if (lp_result.get_dual_solution().size() == (size_t)fixed_problem.n_constraints) {
      warm_primal.resize(lp_result.get_primal_solution().size(), stream);
      raft::copy(
        warm_primal.data(), lp_result.get_primal_solution().data(), warm_primal.size(), stream);
      warm_dual.resize(lp_result.get_dual_solution().size(), stream);
      raft::copy(warm_dual.data(), lp_result.get_dual_solution().data(), warm_dual.size(), stream);
    }
    if (out_state != nullptr &&
        lp_result.get_dual_solution().size() == (size_t)fixed_problem.n_constraints) {
      if (!out_state->valid) { out_state->assignment = in_sol.get_host_assignment(); }
      out_state->primal = cuopt::host_copy(lp_result.get_primal_solution(), stream);
      out_state->dual   = cuopt::host_copy(lp_result.get_dual_solution(), stream);
      out_state->lp_obj = lp_result.get_objective_value();
      out_state->valid  = true;
    }
    raft::copy(fixed_assignment.data(),
               lp_result.get_primal_solution().data(),
               fixed_assignment.size(),
               stream);
    clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
    sol.unfix_variables(fixed_assignment, variable_map);
    bool feas  = sol.get_feasible();
    f_t obj_lp = sol.get_user_objective();
    CUOPT_LOG_INFO(
      "Polish stage %d LP point: feasible=%d obj=%g max_cstr_viol=%g excess=%g elapsed=%.2f",
      (int)k,
      (int)feas,
      obj_lp,
      sol.compute_max_constraint_violation(),
      sol.get_total_excess(),
      timer.elapsed_time());
    if (feas) {
      f_t max_excess = 0;
      if (!strict_point_ok(sol.get_host_assignment(), max_excess) ||
          !original_point_ok(sol.get_host_assignment())) {
        CUOPT_LOG_INFO(
          "Polish stage %d LP point: strict check failed excess=%g", (int)k, max_excess);
        feas = false;
      }
    }
    if (feas) {
      if (obj_lp < best_obj) {
        best_obj = obj_lp;
        improved = true;
        publish_polished(sol, "polish_lp");
      }
      continue;
    }
    // objective-free repair warm-started from the cost-aware point
    auto& lp_state = fixed_problem.lp_state;
    lp_state.resize(fixed_problem, stream);
    thrust::fill(handle_ptr->get_thrust_policy(),
                 lp_state.prev_dual.begin(),
                 lp_state.prev_dual.end(),
                 f_t(0));
    relaxed_lp_settings_t r_settings;
    r_settings.tolerance               = 0.5 * problem_ptr->tolerances.absolute_tolerance;
    r_settings.return_first_feasible   = true;
    r_settings.save_state              = true;
    r_settings.check_infeasibility     = false;
    r_settings.per_constraint_residual = true;
    r_settings.has_initial_primal      = true;
    for (int chunk_id = 0; chunk_id < 3; ++chunk_id) {
      if (ptimer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) {
        break;
      }
      r_settings.time_limit = std::min<double>(20., ptimer.remaining_time());
      if (r_settings.time_limit < 1.) { break; }
      auto resp = get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, r_settings);
      const auto& info = resp.get_additional_termination_information();
      sol.unfix_variables(fixed_assignment, variable_map);
      feas           = sol.get_feasible();
      f_t obj        = sol.get_user_objective();
      f_t max_excess = 0;
      if (feas && (!strict_point_ok(sol.get_host_assignment(), max_excess) ||
                   !original_point_ok(sol.get_host_assignment()))) {
        CUOPT_LOG_INFO("Polish stage %d repair chunk %d: strict check failed excess=%g",
                       (int)k,
                       chunk_id,
                       max_excess);
        feas = false;
        r_settings.tolerance *= 0.1;
      }
      CUOPT_LOG_INFO(
        "Polish stage %d repair chunk %d: status=%s iters=%d feasible=%d obj=%g max_cstr_viol=%g "
        "elapsed=%.2f",
        (int)k,
        chunk_id,
        resp.get_termination_status_string().c_str(),
        info.number_of_steps_taken,
        (int)feas,
        obj,
        sol.compute_max_constraint_violation(),
        timer.elapsed_time());
      if (feas) {
        if (obj < best_obj) {
          best_obj = obj;
          improved = true;
          publish_polished(sol, "polish_repair");
        }
        break;
      }
    }
  }
  CUOPT_LOG_INFO("Polish end: obj0=%g best=%g time=%.2fs elapsed=%.2f",
                 obj0,
                 best_obj,
                 timer.elapsed_time() - t_start,
                 timer.elapsed_time());
  return improved;
}

template <typename i_t, typename f_t>
typename diversity_manager_t<i_t, f_t>::pattern_eval_t
diversity_manager_t<i_t, f_t>::eval_fixed_pattern(const std::vector<f_t>& assign,
                                                  f_t tol,
                                                  f_t tlimit,
                                                  const std::vector<f_t>* warm_primal,
                                                  const std::vector<f_t>* warm_dual,
                                                  bool repair,
                                                  f_t repair_tlimit,
                                                  const char* origin,
                                                  f_t repair_below,
                                                  f_t max_push_time,
                                                  f_t abort_above_res,
                                                  f_t lp_cutoff)
{
  raft::common::nvtx::range fun_scope("eval_fixed_pattern");
  pattern_eval_t res;
  auto* handle_ptr  = problem_ptr->handle_ptr;
  auto stream       = handle_ptr->get_stream();
  const i_t n_cstrs = problem_ptr->n_constraints;
  const f_t abs_tol = problem_ptr->tolerances.absolute_tolerance;
  const double t0   = timer.elapsed_time();
  double full_shift = 0;
  solution_t<i_t, f_t> sol(*problem_ptr);
  sol.copy_new_assignment(assign);
  auto [fixed_problem, fixed_assignment, variable_map] =
    sol.fix_variables(problem_ptr->integer_indices);
  fixed_problem.check_problem_representation(true);
  if (fixed_problem.n_constraints != n_cstrs) { return res; }
  const double t_fixed = timer.elapsed_time();
  double t_lp_start = t_fixed, t_lp_end = t_fixed, t_point = t_fixed, t_check = t_fixed;
  double t_repair = t_fixed, t_publish = t_fixed;
  // a full-length warm primal (all columns of the problem) is gathered to the fixed problem's
  // columns
  std::vector<f_t> warm_fixed;
  if (warm_primal != nullptr && warm_primal->size() == (size_t)problem_ptr->n_variables &&
      warm_primal->size() != (size_t)fixed_problem.n_variables) {
    auto h_map = cuopt::host_copy(variable_map, stream);
    warm_fixed.resize(fixed_problem.n_variables);
    for (i_t i = 0; i < fixed_problem.n_variables; ++i) {
      warm_fixed[i] = (*warm_primal)[h_map[i]];
    }
    warm_primal = &warm_fixed;
  }
  const bool warm = warm_primal != nullptr && warm_dual != nullptr &&
                    warm_primal->size() == (size_t)fixed_problem.n_variables &&
                    warm_dual->size() == (size_t)n_cstrs;
  // the step size of the first fixed LP of this shape (no LP presolve) is reused by the later ones
  auto step_key_ok = [&]() {
    return fixed_lp_step_size > 0 && fixed_lp_step_key[0] == fixed_problem.n_variables &&
           fixed_lp_step_key[1] == fixed_problem.n_constraints &&
           fixed_lp_step_key[2] == fixed_problem.nnz;
  };
  auto use_step = [&](pdlp_solver_settings_t<i_t, f_t>& s) {
    if (s.pdlp_solver_mode == pdlp_solver_mode_t::Stable3 && step_key_ok()) {
      s.set_initial_step_size(fixed_lp_step_size);
    }
  };
  auto keep_step = [&](const pdlp_solver_settings_t<i_t, f_t>& s, auto& lp_result) {
    if (s.pdlp_solver_mode != pdlp_solver_mode_t::Stable3 || step_key_ok()) { return; }
    const f_t st = lp_result.get_pdlp_warm_start_data().initial_step_size_;
    if (!(st > 0) || !std::isfinite(st)) { return; }
    fixed_lp_step_size   = st;
    fixed_lp_step_key[0] = fixed_problem.n_variables;
    fixed_lp_step_key[1] = fixed_problem.n_constraints;
    fixed_lp_step_key[2] = fixed_problem.nnz;
    CUOPT_LOG_INFO("Fixed-pattern LP step size %g kept for the later fixed LPs", (double)st);
  };
  {
    auto snapshot = make_root_lp_snapshot(fixed_problem);
    pdlp_solver_settings_t<i_t, f_t> lp_settings{};
    lp_settings.set_optimality_tolerance(tol);
    lp_settings.time_limit = std::max<double>(1., tlimit);
    lp_settings.method     = method_t::PDLP;
    if (warm) {
      lp_settings.presolver = presolver_t::None;
      use_step(lp_settings);
    }
    std::atomic<int> local_halt{0};
    lp_settings.concurrent_halt = &local_halt;
    lp_settings.inside_mip      = false;
    lp_settings.num_gpus        = context.settings.num_gpus;
    // an infeasible fixed LP (a pattern without the capacity some rows need) stops after a few
    // thousand iterations instead of running into the time limit
    lp_settings.divergence_stop_checks = 3;
    // a fixed LP whose objectives stall (seen as a frozen dual objective with a primal residual of
    // ~10-200 for 100k+ iterations, up to the time limit) ends like at the time limit: the pushes
    // drive its residual down at the same objective
    lp_settings.stall_stop_checks = 3;
    // the fixed problem's user objective lacks the cost of the fixed integers (fixing moves their
    // columns into the row bounds only); full_shift turns it into the full user objective
    const f_t scale       = problem_ptr->presolve_data.objective_scaling_factor;
    const bool stop_above = scale > 0 && (std::isfinite(repair_below) || std::isfinite(lp_cutoff));
    if (stop_above) {
      auto h_obj  = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
      auto h_ints = cuopt::host_copy(problem_ptr->integer_indices, stream);
      double int_obj = 0;
      for (auto j : h_ints) {
        int_obj += (double)h_obj[j] * (double)assign[j];
      }
      full_shift = (double)scale * (int_obj + (double)problem_ptr->presolve_data.objective_offset -
                                    (double)fixed_problem.presolve_data.objective_offset);
    }
    // a fixed LP whose optimum is provably above repair_below (a pattern that can be neither
    // repaired nor published) stops once the Lagrangian bound of a dual iterate, in the full
    // objective, exceeds it. A bound beyond 10x the incumbent is the dual of an infeasible LP
    // growing along its Farkas ray: that LP runs on to the divergence verdict, whose dual is ray-
    // dominated enough for the certificate (at the bound's first crossing it often is not)
    if (stop_above && std::isfinite(repair_below)) {
      const double ray_like = 10. * std::max(1., std::abs((double)repair_below));
      lp_settings.dual_bound_frequency = 1000;
      lp_settings.dual_bound_callback  = [&local_halt, &res, full_shift, repair_below, ray_like](
                                          f_t user_bound) {
        const double b = (double)user_bound + full_shift;
        if (b > res.lp_bound) { res.lp_bound = b; }
        if (b > (double)repair_below && b <= ray_like) { local_halt.store(1); }
      };
    }
    // ... or, warm started, once its primal and dual objectives agree on an optimum clearly above
    // lp_cutoff (a heuristic rule: the dual objective only bounds the optimum for a feasible dual)
    if (warm && stop_above && std::isfinite(lp_cutoff)) {
      lp_settings.objective_cutoff = lp_cutoff - full_shift;
    }
    std::vector<var_t> h_types(fixed_problem.n_variables, var_t::CONTINUOUS);
    snapshot.set_variable_types(h_types.data(), fixed_problem.n_variables);
    if (warm) {
      raft::copy(fixed_assignment.data(), warm_primal->data(), warm_primal->size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(warm_dual->data(), (i_t)warm_dual->size(), stream);
    }
    t_lp_start     = timer.elapsed_time();
    auto lp_result = solve_lp(snapshot, lp_settings);
    t_lp_end       = timer.elapsed_time();
    if (warm) { keep_step(lp_settings, lp_result); }
    const auto& lp_info = lp_result.get_additional_termination_information();
    res.iters           = lp_info.number_of_steps_taken;
    res.l2_primal_res   = lp_info.l2_primal_residual;
    res.optimal         = lp_result.get_termination_status() == pdlp_termination_status_t::Optimal;
    res.infeasible =
      lp_result.get_termination_status() == pdlp_termination_status_t::PrimalInfeasible;
    // 1: stopped on the Lagrangian bound (valid), 2: on the primal/dual agreement rule
    if (lp_result.get_termination_status() == pdlp_termination_status_t::ConcurrentLimit &&
        res.lp_bound > (double)repair_below) {
      res.cutoff = 1;
    } else if (std::isfinite(lp_settings.objective_cutoff) &&
               lp_result.get_termination_status() == pdlp_termination_status_t::IterationLimit) {
      res.cutoff = 2;
    }
    if (res.cutoff) {
      CUOPT_LOG_INFO(
        "Fixed-pattern LP cutoff (%s): rule=%s iters=%d lagrangian_bound=%g primal_obj=%g "
        "dual_obj=%g rel_primal_res=%g rel_dual_res=%g cutoff=%g lp_time=%.2fs time_limit=%.0fs",
        origin,
        res.cutoff == 1 ? "bound" : "pd",
        lp_info.number_of_steps_taken,
        res.lp_bound,
        (double)lp_info.primal_objective + full_shift,
        (double)lp_info.dual_objective + full_shift,
        (double)lp_info.l2_relative_primal_residual,
        (double)lp_info.l2_relative_dual_residual,
        (double)(std::isfinite(lp_cutoff) ? lp_cutoff : repair_below),
        timer.elapsed_time() - t0,
        (double)lp_settings.time_limit);
    }
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
  res.lp_time = timer.elapsed_time() - t0;
  clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
  sol.unfix_variables(fixed_assignment, variable_map);
  // full objective (fixed integers included) of the (slightly infeasible) LP point
  res.lp_obj        = sol.get_user_objective();
  res.usable        = std::isfinite(res.lp_obj);
  res.lp_assignment = sol.get_host_assignment();
  res.aborted       = !res.optimal && res.l2_primal_res > abort_above_res;
  t_point           = timer.elapsed_time();
  if (repair && res.usable && res.lp_obj < repair_below && !res.aborted && !res.infeasible &&
      !res.cutoff) {
    // phase B: the PID point's l2 primal residual stalls ~50-100 (repair +2%); warm-started
    // pushes with a frozen large primal weight drive it to ~1e-3 at a nearly unchanged objective
    // (the ranking objective above stays the phase-A one)
    const double t_push = timer.elapsed_time();
    // each push ends at its time cap or once its l2 primal residual reaches its target: the first
    // cuts the residual by orders of magnitude within ~1000 iterations and then creeps (~0.07 to
    // ~0.03 over its last 10k iterations), the second reaches 5e-3 within ~1000 iterations; the
    // repair needs about the same time from 5e-3 as from 5e-4
    struct push_t {
      double weight;
      double cap;
      double target;
    };
    const double push_scale = std::clamp<double>(max_push_time / 11., 0., 1.);
    const push_t pushes[]   = {{1e4, 6. * push_scale, 5e-2}, {1e5, 5. * push_scale, 5e-3}};
    auto cur_p              = cuopt::device_copy(res.primal, stream);
    auto cur_d              = cuopt::device_copy(res.dual, stream);
    bool pushed             = false;
    // a push that leaves the primal residual where the PID point had it (instead of cutting it by
    // orders of magnitude) means the fixed LP is infeasible: no further push, no repair
    auto push_stalled = [&]() {
      return pushed && res.pushed_l2_primal_res > 1. &&
             res.pushed_l2_primal_res > 0.5 * res.l2_primal_res;
    };
    for (const auto& ps : pushes) {
      if (timer.check_time_limit() || check_b_b_preemption() || timer.remaining_time() < 30.) {
        break;
      }
      auto snapshot = make_root_lp_snapshot(fixed_problem);
      pdlp_solver_settings_t<i_t, f_t> lp_settings{};
      lp_settings.set_optimality_tolerance(1e-7);
      lp_settings.tolerances.absolute_primal_tolerance = ps.target;
      lp_settings.tolerances.relative_primal_tolerance = 0;
      lp_settings.first_primal_feasible                = true;
      lp_settings.time_limit                           = ps.cap;
      lp_settings.method     = method_t::PDLP;
      lp_settings.presolver  = presolver_t::None;
      std::atomic<int> local_halt{0};
      lp_settings.concurrent_halt = &local_halt;
      lp_settings.inside_mip      = false;
      lp_settings.num_gpus        = context.settings.num_gpus;
      std::vector<var_t> h_types(fixed_problem.n_variables, var_t::CONTINUOUS);
      snapshot.set_variable_types(h_types.data(), fixed_problem.n_variables);
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
      use_step(lp_settings);
      auto lp_result = solve_lp(snapshot, lp_settings);
      keep_step(lp_settings, lp_result);
      if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables ||
          lp_result.get_dual_solution().size() != (size_t)n_cstrs) {
        break;
      }
      res.pushed_l2_primal_res =
        lp_result.get_additional_termination_information().l2_primal_residual;
      raft::copy(cur_p.data(), lp_result.get_primal_solution().data(), cur_p.size(), stream);
      raft::copy(cur_d.data(), lp_result.get_dual_solution().data(), cur_d.size(), stream);
      pushed = true;
      if (push_stalled()) { break; }
    }
    if (pushed) {
      raft::copy(fixed_assignment.data(), cur_p.data(), cur_p.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      sol.unfix_variables(fixed_assignment, variable_map);
      res.pushed_lp_obj     = sol.get_user_objective();
      res.pushed_assignment = sol.get_host_assignment();
    }
    res.push_time         = timer.elapsed_time() - t_push;
    const double t_repair0 = timer.elapsed_time();
    bool feas              = sol.get_feasible();
    if (feas) {
      f_t max_excess = 0;
      feas           = strict_point_ok(sol.get_host_assignment(), max_excess) &&
             original_point_ok(sol.get_host_assignment());
    }
    t_check = timer.elapsed_time();
    if (!feas && !push_stalled()) {
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
        if (feas) {
          f_t max_excess = 0;
          if (!strict_point_ok(sol.get_host_assignment(), max_excess)) {
            CUOPT_LOG_INFO("Pattern repair: strict check failed excess=%g", max_excess);
            feas = false;
            r_settings.tolerance *= 0.1;
          } else if (!original_point_ok(sol.get_host_assignment())) {
            CUOPT_LOG_INFO("Pattern repair: original-space check failed");
            feas = false;
            r_settings.tolerance *= 0.1;
          }
        }
      }
    }
    t_repair        = timer.elapsed_time();
    res.repair_time = t_repair - t_repair0;
    if (feas) {
      res.repaired_obj        = sol.get_user_objective();
      res.repaired_assignment = sol.get_host_assignment();
      const f_t best_pub      = population.is_feasible()
                                  ? population.best_feasible().get_user_objective()
                                  : std::numeric_limits<f_t>::infinity();
      // feas holds only for a point that passed the strict and original-space checks above
      if (res.repaired_obj < best_pub) {
        res.published = true;
        population.add_solution(solution_t<i_t, f_t>(sol), origin);
      }
    }
    t_publish = timer.elapsed_time();
  }
  res.time = timer.elapsed_time() - t0;
  CUOPT_LOG_INFO(
    "Pattern eval timing: fix=%.2f lp_setup=%.2f lp=%.2f point=%.2f push=%.2f check=%.2f "
    "repair=%.2f publish=%.2f total=%.2f",
    t_fixed - t0,
    t_lp_start - t_fixed,
    t_lp_end - t_lp_start,
    t_point - t_lp_end,
    res.push_time,
    std::max(0., t_check - t_point - res.push_time),
    std::max(0., t_repair - t_check),
    std::max(0., t_publish - t_repair),
    res.time);
  return res;
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::strict_point_ok(const std::vector<f_t>& x, f_t& max_excess)
{
  auto* handle_ptr    = problem_ptr->handle_ptr;
  auto stream         = handle_ptr->get_stream();
  const i_t n_vars    = problem_ptr->n_variables;
  const i_t n_cstrs   = problem_ptr->n_constraints;
  const f_t abs_tol   = problem_ptr->tolerances.absolute_tolerance;
  auto h_offsets      = cuopt::host_copy(problem_ptr->offsets, stream);
  auto h_cols         = cuopt::host_copy(problem_ptr->variables, stream);
  auto h_vals         = cuopt::host_copy(problem_ptr->coefficients, stream);
  auto h_clb          = cuopt::host_copy(problem_ptr->constraint_lower_bounds, stream);
  auto h_cub          = cuopt::host_copy(problem_ptr->constraint_upper_bounds, stream);
  auto [h_vlb, h_vub] = cuopt::extract_host_bounds<f_t>(problem_ptr->variable_bounds, handle_ptr);
  max_excess          = 0;
  if ((i_t)x.size() != n_vars) {
    max_excess = std::numeric_limits<f_t>::infinity();
    return false;
  }
  for (i_t j = 0; j < n_vars; ++j) {
    if (!std::isfinite(x[j])) {
      max_excess = std::numeric_limits<f_t>::infinity();
      return false;
    }
    f_t ex = std::max(h_vlb[j] - x[j], x[j] - h_vub[j]);
    if (ex > max_excess) max_excess = ex;
  }
  for (i_t r = 0; r < n_cstrs; ++r) {
    // Neumaier-compensated sum of products (fma error term folded in)
    double sum = 0, comp = 0;
    for (i_t k = h_offsets[r]; k < h_offsets[r + 1]; ++k) {
      const double p   = (double)h_vals[k] * (double)x[h_cols[k]];
      const double err = std::fma((double)h_vals[k], (double)x[h_cols[k]], -p);
      const double t   = sum + p;
      comp += std::abs(sum) >= std::abs(p) ? (sum - t) + p : (p - t) + sum;
      comp += err;
      sum = t;
    }
    const f_t a  = (f_t)(sum + comp);
    const f_t lo = h_clb[r], hi = h_cub[r];
    if (std::isfinite(lo)) {
      f_t tol    = 1e-12 * std::max<f_t>(1., std::max(std::abs(a), std::abs(lo)));
      max_excess = std::max(max_excess, lo - a - tol);
    }
    if (std::isfinite(hi)) {
      f_t tol    = 1e-12 * std::max<f_t>(1., std::max(std::abs(a), std::abs(hi)));
      max_excess = std::max(max_excess, a - hi - tol);
    }
  }
  return max_excess <= 0.5 * abs_tol;
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::original_point_ok(const std::vector<f_t>& x)
{
  if (!problem_ptr->has_papilo_presolve_data()) { return true; }
  const auto* presolver = problem_ptr->presolve_data.papilo_presolve_ptr;
  if (presolver == nullptr) { return true; }
  auto* handle_ptr = problem_ptr->handle_ptr;
  auto stream      = handle_ptr->get_stream();
  rmm::device_uvector<f_t> d_x(x.size(), stream);
  raft::copy(d_x.data(), x.data(), x.size(), stream);
  // private copy of the presolve data: post-processing writes into its fixed-variable buffer,
  // which the publication path of other threads uses on the shared problem
  presolve_data_t<i_t, f_t> pd(problem_ptr->presolve_data, stream);
  const double t0 = timer.elapsed_time();
  pd.post_process_assignment(*problem_ptr, d_x, true, stream);
  pd.papilo_uncrush_assignment(d_x, stream);
  auto h_x          = cuopt::host_copy(d_x, stream);
  const double t1   = timer.elapsed_time();
  const f_t abs_tol = problem_ptr->tolerances.absolute_tolerance;
  const bool ok =
    presolver->is_original_primal_solution_feasible(h_x, 0.5 * abs_tol, 0., 0.5 * abs_tol);
  CUOPT_LOG_INFO("Original-space check: uncrush=%.2fs check=%.2fs ok=%d",
                 t1 - t0,
                 timer.elapsed_time() - t1,
                 (int)ok);
  return ok;
}

namespace {

template <typename i_t, typename f_t>
struct slope_link_t {
  i_t row;
  i_t var;
  i_t pos;    // nnz position of the binary's coefficient
  f_t cap;    // |coefficient|: usage <= cap * y
  f_t sign;   // usage = sign * (cont activity) - rhs_term
  f_t rhs;    // usage = sign * act - rhs
  bool pair;  // one of the two binaries of a pair row
};

template <typename i_t, typename f_t>
struct slope_links_t {
  std::vector<slope_link_t<i_t, f_t>> links;
  std::vector<i_t> n_links;
  // the links of the binaries with exactly one link: the ones slope scaling scales
  std::vector<slope_link_t<i_t, f_t>> vub;
  i_t n_single_int_rows{0};
  i_t n_single_int_other{0};
  i_t n_pair_rows{0};
};

// Rows with one or two integers (binaries) and at least one continuous column, one-sided, where
// each binary at 1 relaxes the row (pair rows: usage <= cap1 * y1 + cap2 * y2).
template <typename i_t, typename f_t>
slope_links_t<i_t, f_t> detect_slope_links(i_t n_vars,
                                           i_t n_cstrs,
                                           const std::vector<i_t>& h_offsets,
                                           const std::vector<i_t>& h_cols,
                                           const std::vector<f_t>& h_vals,
                                           const std::vector<f_t>& h_clb,
                                           const std::vector<f_t>& h_cub,
                                           const std::vector<f_t>& h_vlb,
                                           const std::vector<f_t>& h_vub,
                                           const std::vector<char>& is_int)
{
  slope_links_t<i_t, f_t> det;
  auto& links             = det.links;
  auto& n_links           = det.n_links;
  auto& n_single_int_rows = det.n_single_int_rows;
  auto& n_single_int_other = det.n_single_int_other;
  auto& n_pair_rows       = det.n_pair_rows;
  n_links.assign(n_vars, 0);
  for (i_t r = 0; r < n_cstrs; ++r) {
    i_t n_i = 0, n_c = 0, pos = -1, pos2 = -1;
    for (i_t k = h_offsets[r]; k < h_offsets[r + 1]; ++k) {
      if (is_int[h_cols[k]]) {
        n_i++;
        pos2 = pos;
        pos  = k;
      } else {
        n_c++;
      }
    }
    if (n_i == 2 && n_c > 0) {
      const i_t y1 = h_cols[pos2], y2 = h_cols[pos];
      const f_t b1 = h_vals[pos2], b2 = h_vals[pos];
      const bool bin = h_vlb[y1] == 0 && h_vub[y1] == 1 && h_vlb[y2] == 0 && h_vub[y2] == 1;
      const bool le  = std::isfinite(h_cub[r]) && !std::isfinite(h_clb[r]);
      const bool ge  = std::isfinite(h_clb[r]) && !std::isfinite(h_cub[r]);
      if (bin && le && b1 < 0 && b2 < 0) {
        links.push_back({r, y1, pos2, -b1, f_t(1), h_cub[r], true});
        links.push_back({r, y2, pos, -b2, f_t(1), h_cub[r], true});
      } else if (bin && ge && b1 > 0 && b2 > 0) {
        links.push_back({r, y1, pos2, b1, f_t(-1), -h_clb[r], true});
        links.push_back({r, y2, pos, b2, f_t(-1), -h_clb[r], true});
      } else {
        continue;
      }
      n_links[y1]++;
      n_links[y2]++;
      n_pair_rows++;
      continue;
    }
    if (n_i != 1 || n_c == 0) { continue; }
    n_single_int_rows++;
    const i_t y       = h_cols[pos];
    const f_t b       = h_vals[pos];
    const bool binary = h_vlb[y] == 0 && h_vub[y] == 1;
    const bool le     = std::isfinite(h_cub[r]) && !std::isfinite(h_clb[r]);
    const bool ge     = std::isfinite(h_clb[r]) && !std::isfinite(h_cub[r]);
    if (binary && le && b < 0) {
      links.push_back({r, y, pos, -b, f_t(1), h_cub[r], false});
      n_links[y]++;
    } else if (binary && ge && b > 0) {
      links.push_back({r, y, pos, b, f_t(-1), -h_clb[r], false});
      n_links[y]++;
    } else {
      n_single_int_other++;
    }
  }
  // binaries with exactly one link are scaled
  for (size_t q = 0; q < links.size(); ++q) {
    const auto& l = links[q];
    if (n_links[l.var] != 1) { continue; }
    if (l.pair) {
      const auto& o =
        (q + 1 < links.size() && links[q + 1].row == l.row) ? links[q + 1] : links[q - 1];
      if (n_links[o.var] != 1) { continue; }
    }
    det.vub.push_back(l);
  }
  return det;
}

}  // namespace

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::has_slope_scaling_links()
{
  if (problem_ptr->n_integer_vars == 0 || problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return false;
  }
  auto* handle_ptr    = problem_ptr->handle_ptr;
  auto stream         = handle_ptr->get_stream();
  auto h_offsets      = cuopt::host_copy(problem_ptr->offsets, stream);
  auto h_cols         = cuopt::host_copy(problem_ptr->variables, stream);
  auto h_vals         = cuopt::host_copy(problem_ptr->coefficients, stream);
  auto h_clb          = cuopt::host_copy(problem_ptr->constraint_lower_bounds, stream);
  auto h_cub          = cuopt::host_copy(problem_ptr->constraint_upper_bounds, stream);
  auto h_int          = cuopt::host_copy(problem_ptr->integer_indices, stream);
  auto [h_vlb, h_vub] = cuopt::extract_host_bounds<f_t>(problem_ptr->variable_bounds, handle_ptr);
  std::vector<char> is_int(problem_ptr->n_variables, 0);
  for (auto v : h_int) {
    is_int[v] = 1;
  }
  return !detect_slope_links<i_t, f_t>(problem_ptr->n_variables,
                                       problem_ptr->n_constraints,
                                       h_offsets,
                                       h_cols,
                                       h_vals,
                                       h_clb,
                                       h_cub,
                                       h_vlb,
                                       h_vub,
                                       is_int)
            .vub.empty();
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_slope_scaling()
{
  raft::common::nvtx::range fun_scope("run_slope_scaling");
  if (slope_scaling_done || problem_ptr->n_integer_vars == 0 ||
      problem_ptr->n_integer_vars == problem_ptr->n_variables || !ls.lp_optimal_exists) {
    return false;
  }
  slope_scaling_done = true;
  auto* handle_ptr   = problem_ptr->handle_ptr;
  auto stream        = handle_ptr->get_stream();
  const f_t budget   = std::min<f_t>(std::min<f_t>(400., 0.25 * context.settings.time_limit),
                                   timer.remaining_time() * 0.4);
  if (budget < 30.) { return false; }
  const double t0     = timer.elapsed_time();
  const i_t n_vars    = problem_ptr->n_variables;
  const i_t n_cstrs   = problem_ptr->n_constraints;
  const f_t abs_tol   = problem_ptr->tolerances.absolute_tolerance;
  const f_t int_tol   = problem_ptr->tolerances.integrality_tolerance;
  auto h_offsets      = cuopt::host_copy(problem_ptr->offsets, stream);
  auto h_cols         = cuopt::host_copy(problem_ptr->variables, stream);
  auto h_vals         = cuopt::host_copy(problem_ptr->coefficients, stream);
  auto h_clb          = cuopt::host_copy(problem_ptr->constraint_lower_bounds, stream);
  auto h_cub          = cuopt::host_copy(problem_ptr->constraint_upper_bounds, stream);
  auto h_int          = cuopt::host_copy(problem_ptr->integer_indices, stream);
  auto h_obj          = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
  auto [h_vlb, h_vub] = cuopt::extract_host_bounds<f_t>(problem_ptr->variable_bounds, handle_ptr);
  std::vector<char> is_int(n_vars, 0);
  for (auto v : h_int) {
    is_int[v] = 1;
  }

  // ---- detection
  auto det = detect_slope_links<i_t, f_t>(
    n_vars, n_cstrs, h_offsets, h_cols, h_vals, h_clb, h_cub, h_vlb, h_vub, is_int);
  const auto& links              = det.links;
  const auto& n_links            = det.n_links;
  const auto& vub                = det.vub;
  const i_t n_single_int_rows    = det.n_single_int_rows;
  const i_t n_single_int_other   = det.n_single_int_other;
  const i_t n_pair_rows          = det.n_pair_rows;
  i_t n_pair_links = 0;
  for (auto& l : vub) {
    n_pair_links += l.pair;
  }
  i_t n_bins = 0, n_multi = 0, n_bin_cost = 0;
  for (auto v : h_int) {
    if (h_vlb[v] == 0 && h_vub[v] == 1) { n_bins++; }
    if (n_links[v] > 1) { n_multi++; }
  }
  // other rows the linked binaries appear in (coupling: objective and resource rows)
  std::vector<char> is_link_row(n_cstrs, 0), is_vub_bin(n_vars, 0);
  for (auto& l : vub) {
    is_link_row[l.row] = 1;
    is_vub_bin[l.var]  = 1;
    if (h_obj[l.var] != 0) { n_bin_cost++; }
  }
  int64_t n_other_nnz = 0;
  i_t n_other_rows    = 0;
  // one-sided rows (not link rows) holding linked binaries and continuous columns: the binaries
  // take a share of the row (e.g. a capacity the flows share with the binaries' fixed use)
  std::vector<i_t> mixed_coupling_rows;
  for (i_t r = 0; r < n_cstrs; ++r) {
    if (is_link_row[r]) { continue; }
    bool has = false, has_cont = false;
    for (i_t k = h_offsets[r]; k < h_offsets[r + 1]; ++k) {
      if (is_vub_bin[h_cols[k]]) {
        has = true;
        n_other_nnz++;
      }
      has_cont = has_cont || !is_int[h_cols[k]];
    }
    n_other_rows += has;
    if (has && has_cont && std::isfinite(h_clb[r]) != std::isfinite(h_cub[r])) {
      mixed_coupling_rows.push_back(r);
    }
  }
  std::vector<f_t> caps;
  for (auto& l : vub) {
    caps.push_back(l.cap);
  }
  std::sort(caps.begin(), caps.end());
  auto q = [&](double p) { return caps.empty() ? f_t(0) : caps[(size_t)(p * (caps.size() - 1))]; };
  CUOPT_LOG_INFO(
    "Slope scaling detect: binaries=%d single_int_rows=%d link_rows=%zu other_single_int=%d "
    "linked_binaries=%zu multi_link_binaries=%d linked_with_obj_cost=%d coupling_rows=%d "
    "coupling_nnz=%ld cap_q(0,.5,1)=%g/%g/%g time=%.2fs elapsed=%.2f",
    n_bins,
    n_single_int_rows,
    links.size(),
    n_single_int_other,
    vub.size(),
    n_multi,
    n_bin_cost,
    n_other_rows,
    (long)n_other_nnz,
    q(0),
    q(0.5),
    q(1),
    timer.elapsed_time() - t0,
    timer.elapsed_time());
  CUOPT_LOG_INFO("Slope scaling detect: pair_rows=%d pair_links=%d single_links=%d",
                 n_pair_rows,
                 n_pair_links,
                 (int)vub.size() - n_pair_links);
  if (vub.empty()) { return false; }
  // the rounds and pattern LPs take the GPU from the B&B pre-basis PDLP cut loop (which only
  // raises the bound) until slope scaling ends
  auto* bnb = context.branch_and_bound_ptr;
  if (bnb != nullptr) { bnb->set_pdlp_cut_yield(true); }
  cuopt::scope_guard release_gpu([bnb]() {
    if (bnb != nullptr) { bnb->set_pdlp_cut_yield(false); }
  });

  // ---- slope-scaling rounds
  auto cur_x     = cuopt::host_copy(lp_optimal_solution, stream);
  auto cur_y     = cuopt::host_copy(lp_dual_optimal_solution, stream);
  const size_t L = vub.size();
  std::vector<f_t> scale(L);  // current per-link usage scale (cap at the root LP)
  for (size_t k = 0; k < L; ++k) {
    scale[k] = vub[k].cap;
  }
  auto usage = [&](size_t k, const std::vector<f_t>& x) {
    const auto& l = vub[k];
    double act    = 0;
    for (i_t p = h_offsets[l.row]; p < h_offsets[l.row + 1]; ++p) {
      if (is_int[h_cols[p]]) { continue; }
      act += (double)h_vals[p] * (double)x[h_cols[p]];
    }
    return (f_t)(l.sign * act - l.rhs);
  };
  constexpr f_t open_rel  = 1e-4;  // usage above open_rel * cap counts as open
  constexpr f_t floor_rel = 1e-3;  // scale never below floor_rel * cap
  auto open_tol = [&](size_t k) { return std::max<f_t>(open_rel * vub[k].cap, 10 * abs_tol); };
  std::vector<f_t> u(L);
  std::vector<char> is_open(L, 0), prev_open(L, 0);
  for (size_t k = 0; k < L; ++k) {
    u[k]         = usage(k, cur_x);
    prev_open[k] = u[k] > open_tol(k);
  }
  {
    i_t n_open = 0;
    for (size_t k = 0; k < L; ++k) {
      n_open += prev_open[k];
    }
    CUOPT_LOG_INFO(
      "Slope scaling root point: open=%d of %zu elapsed=%.2f", n_open, L, timer.elapsed_time());
  }
  // fix-and-propagate engine for the patterns
  host_fix_propagate_t<i_t, f_t> fp_host;
  fp_host.abs_tol = abs_tol;
  fp_host.continuous_min_improvement =
    context.settings.heuristic_params.fix_propagate_continuous_min_improvement;
  fp_host.build(n_vars, n_cstrs, h_offsets, h_cols, h_vals, h_clb, h_cub, h_vlb, h_vub, h_int);
  fp_host.int_tol = int_tol;
  {
    simplex::user_problem_t<i_t, f_t> user_problem(problem_ptr->handle_ptr);
    problem_ptr->get_host_user_problem(user_problem);
    fp_host.build_domain(user_problem, problem_ptr->clique_table.get());
  }
  std::vector<i_t> link_of(n_vars, -1);
  for (size_t k = 0; k < L; ++k) {
    link_of[vub[k].var] = (i_t)k;
  }
  // the other link of a pair row (-1 for single links); the two links of a pair row are adjacent
  std::vector<i_t> pair_partner(L, -1);
  for (size_t k = 0; k < L; ++k) {
    if (!vub[k].pair) { continue; }
    if (k + 1 < L && vub[k + 1].row == vub[k].row) {
      pair_partner[k] = (i_t)k + 1;
    } else if (k > 0 && vub[k - 1].row == vub[k].row) {
      pair_partner[k] = (i_t)k - 1;
    }
  }
  // binary-only rows (every column a binary) and, per binary, the binary-only rows holding it
  // (position in brows, coefficient): the preference repair of the patterns works on them
  std::vector<i_t> brows;
  std::vector<i_t> bvar_off(n_vars + 1, 0), bvar_row;
  std::vector<f_t> bvar_val;
  {
    auto is_bin = [&](i_t j) { return is_int[j] && h_vlb[j] == 0 && h_vub[j] == 1; };
    for (i_t r = 0; r < n_cstrs; ++r) {
      if (h_offsets[r + 1] == h_offsets[r]) { continue; }
      bool all_bin = true;
      for (i_t p = h_offsets[r]; p < h_offsets[r + 1] && all_bin; ++p) {
        all_bin = is_bin(h_cols[p]);
      }
      if (!all_bin) { continue; }
      brows.push_back(r);
      for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
        bvar_off[h_cols[p] + 1]++;
      }
    }
    for (i_t j = 0; j < n_vars; ++j) {
      bvar_off[j + 1] += bvar_off[j];
    }
    bvar_row.resize(bvar_off[n_vars]);
    bvar_val.resize(bvar_off[n_vars]);
    std::vector<i_t> fill(bvar_off.begin(), bvar_off.end() - 1);
    for (size_t q = 0; q < brows.size(); ++q) {
      const i_t r = brows[q];
      for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
        bvar_row[fill[h_cols[p]]]   = (i_t)q;
        bvar_val[fill[h_cols[p]]++] = h_vals[p];
      }
    }
  }
  // links by continuous column (CSR): the links whose row holds the column, i.e. whose binary
  // gives that flow its capacity
  std::vector<i_t> col_link_off(n_vars + 1, 0), col_link;
  for (size_t k = 0; k < L; ++k) {
    for (i_t p = h_offsets[vub[k].row]; p < h_offsets[vub[k].row + 1]; ++p) {
      if (!is_int[h_cols[p]]) { col_link_off[h_cols[p] + 1]++; }
    }
  }
  for (i_t j = 0; j < n_vars; ++j) {
    col_link_off[j + 1] += col_link_off[j];
  }
  col_link.resize(col_link_off[n_vars]);
  {
    std::vector<i_t> fill(col_link_off.begin(), col_link_off.end() - 1);
    for (size_t k = 0; k < L; ++k) {
      for (i_t p = h_offsets[vub[k].row]; p < h_offsets[vub[k].row + 1]; ++p) {
        if (!is_int[h_cols[p]]) { col_link[fill[h_cols[p]]++] = (i_t)k; }
      }
    }
  }
  // phase 1 ends at t0 + budget; once a pattern was published, or from the start when pair rows
  // are scaled (their patterns need more rounds), the rounds and evaluations go on up to
  // t0 + ext_budget
  // t0 + ext_budget, end_margin before the time limit: left to the publication of the last
  // result and the solver's shutdown
  constexpr f_t end_margin = 20.;
  bool published           = false;
  const bool extended      = n_pair_links > 0;
  const f_t ext_budget     = std::max<f_t>(budget, timer.remaining_time() - end_margin);
  auto remaining           = [&]() {
    return t0 + (published || extended ? ext_budget : budget) - timer.elapsed_time();
  };
  auto stop = [&]() {
    return remaining() <= 0 || timer.check_time_limit() || check_b_b_preemption() ||
           timer.remaining_time() < end_margin;
  };
  std::vector<std::vector<f_t>> seen;
  int n_evals = 0;
  std::vector<double> eval_times;  // durations of the pattern evaluations, for the end rule
  // binaries opened by a flip repair, for the retry of the same point only
  std::vector<char> forced_open(n_vars, 0);
  i_t n_forced      = 0;
  // integers moved by the certificate repair of an infeasible fixed LP (value, NaN = none), fixed
  // first in the retries of the same point
  const f_t no_val = std::numeric_limits<f_t>::quiet_NaN();
  auto has_val     = [](f_t v) { return !std::isnan(v); };
  std::vector<f_t> cert_val(n_vars, no_val);
  i_t n_cert                   = 0;
  constexpr i_t max_cert_moves = 100;
  // infeasibility certificates learned from the infeasible fixed LPs of this run: the Farkas ray of
  // a pattern LP with its z_j on the integers and D0, its objective without the integer terms. The
  // ray depends on the rows and the continuous bounds only, so any integer pattern v with
  // D0 + sum z_j v_j > 0 is infeasible by the same ray (a feasibility cut): every pattern is
  // checked against them after fix-and-propagate and moved before it spends an LP (a retry whose
  // propagation re-opened a consumer of the deficit row, a later round's pattern on the same
  // deficit rows)
  struct ray_t {
    std::vector<std::pair<i_t, double>> z;
    double D0;
    int hits;
  };
  std::vector<ray_t> rays;
  constexpr size_t max_rays = 16;
  // mode-0 fixed LPs evaluated for the current point (its later ones are retries)
  int point_lps = 0;
  // moves of a ray at the integer values val (most cancelling first, until twice D is cancelled)
  auto ray_moves = [&](const std::vector<std::pair<i_t, double>>& z_int,
                       auto&& val,
                       double D,
                       i_t& n_moved,
                       double& cancelled) {
    std::vector<std::pair<double, std::pair<i_t, f_t>>> mv;
    for (const auto& [j, z] : z_int) {
      if (has_val(cert_val[j])) { continue; }
      const double v  = val(j);
      const bool unit = h_vub[j] - h_vlb[j] <= 1;
      if (z < 0 && v < h_vub[j] - 0.5) {
        const f_t t = unit ? h_vub[j] : (f_t)(v + 1);
        mv.push_back({-z * ((double)t - v), {j, t}});
      } else if (z > 0 && v > h_vlb[j] + 0.5) {
        const f_t t = unit ? h_vlb[j] : (f_t)(v - 1);
        mv.push_back({z * (v - (double)t), {j, t}});
      }
    }
    std::sort(mv.begin(), mv.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::pair<i_t, f_t>> taken;
    for (const auto& [g, jt] : mv) {
      if (n_cert + n_moved >= max_cert_moves ||
          (n_moved > 0 && (D > 0 ? cancelled >= 2 * D : n_moved >= 10))) {
        break;
      }
      cert_val[jt.first] = jt.second;
      taken.push_back(jt);
      n_moved++;
      cancelled += g;
    }
    return taken;
  };
  // conflict promotions learned by the patterns of this run (variable, value it was fixed at): the
  // same closed binaries of infeasible clusters conflict in most patterns, so every pattern starts
  // with the promotions of the earlier ones instead of rediscovering them one attempt at a time
  std::vector<std::pair<i_t, f_t>> learned;
  std::vector<char> is_learned(n_vars, 0);
  f_t last_eval_obj = std::numeric_limits<f_t>::infinity();  // repaired objective of the last eval
  // hysteresis of the mode-0 patterns after a publish: the linked binaries open in the last
  // published point (anchor_open) stay open while their y is at least keep_frac * thr, the
  // others need thr; later patterns are then moves from the incumbent's design (closing every
  // used link below thr made most of them LP-infeasible or conflicting)
  std::vector<char> anchor_open(n_vars, 0);
  std::vector<f_t> anchor_u(L, 0);  // link row usage at the last published point
  constexpr f_t keep_frac = 0.5;
  // phase-A l2 primal residual of the last mode-0 first attempt (bound for its retry)
  f_t first_try_res = std::numeric_limits<f_t>::infinity();
  f_t anchor_obj    = std::numeric_limits<f_t>::infinity();  // objective of the anchored point
  // the anchored (last published) point and its fixed-LP duals: the point of the neighbourhood
  // evaluations (mode 1 and 2), kept while the rounds move cur_x / cur_y
  std::vector<f_t> anchor_x, anchor_y;
  // near miss: a round pattern whose fixed LP ends just above the incumbent (within near_frac)
  // is repaired anyway; its point (and fixed-LP duals) is the anchor of one safe chain, whose
  // first steps gain 1-2% after a round pattern (late round patterns end 0.9-1.4% above)
  constexpr f_t near_frac = f_t(0.015);
  f_t near_obj            = std::numeric_limits<f_t>::infinity();
  std::vector<f_t> near_x, near_y;
  // mode 2 pattern (neighbourhood of the anchored point): preferred value and order key per
  // integer, set by the caller
  std::vector<f_t> nb_pref(n_vars, 0), nb_key(n_vars, 0);
  // rows with a continuous column (in the numbering of fp_host): the only rows whose duals the
  // fixed LP determines
  std::vector<char> frow_mixed(fp_host.n_rows(), 0);
  for (i_t r = 0; r < fp_host.n_rows(); ++r) {
    for (i_t p = fp_host.row_offsets[r]; p < fp_host.row_offsets[r + 1]; ++p) {
      if (!is_int[fp_host.row_cols[p]]) {
        frow_mixed[r] = 1;
        break;
      }
    }
  }
  // number of rows per column (a continuous column of a single row absorbs that row's changes)
  std::vector<i_t> col_nrows(n_vars, 0);
  for (auto j : h_cols) {
    col_nrows[j]++;
  }
  // the links of the used-link pattern of the anchored point (mode 1, and the base of the flip
  // steps): the links carrying flow, and the open links whose closing would take the point out of
  // another of its rows: an integer-only row (closing it would make fix-and-propagate move other
  // integers, e.g. open the other binary of a cover), or a row holding continuous columns (a row
  // the binary shares with flows) that no single-row continuous column absorbs. The anchored
  // point then stays feasible for the pattern's fixed LP.
  std::vector<char> link_keep(n_vars, 0);
  i_t n_keep_used = 0, n_keep_rows = 0, n_keep_int_rows = 0, n_close = 0;
  // the anchored point as moved by the accepted moves: row activities (fp_host numbering) and the
  // values of the single-row continuous columns that absorb them
  std::vector<double> mv_act;
  std::vector<f_t> mv_x;
  struct move_t {
    i_t col, row;
    double coef, value, old;
  };
  std::vector<move_t> mv_cols;
  bool safe_warm = false;  // the pattern's fixed LP starts from the moved point (a safe step)
  i_t mv_block   = -1;  // the row (fp_host numbering) that blocked the last move refused
  // moves integer v by d at the moved point unless that takes the point out of one of v's rows
  // (other than skip_row): 0 moved, 1 an integer-only row, 2 a row with continuous columns that
  // no single-row continuous column absorbs. With viol_int, the integer-only rows the move takes
  // out are listed there instead of refusing it (the first move of a compound move). A row already
  // out (inside a compound move) may stay out if the move does not take it further.
  auto try_move = [&](i_t v, double d, i_t skip_row, std::vector<i_t>* viol_int) -> int {
    mv_cols.clear();
    mv_block = -1;
    if (viol_int != nullptr) { viol_int->clear(); }
    for (i_t kk = fp_host.var_offsets[v]; kk < fp_host.var_offsets[v + 1]; ++kk) {
      const i_t r = fp_host.var_rows[kk];
      if (fp_host.row_orig[r] == skip_row) { continue; }
      const double a0 = mv_act[r];
      const double na = a0 + (double)fp_host.var_vals[kk] * d;
      const double lo = fp_host.row_lb[r], hi = fp_host.row_ub[r];
      const double tol =
        1e-6 * std::max({1., std::isfinite(lo) ? std::abs(lo) : 0., std::isfinite(hi) ? std::abs(hi) : 0.});
      if (na <= hi + tol && na >= lo - tol) { continue; }
      const double out0 = std::max({0., a0 - hi, lo - a0});
      const double out1 = std::max({0., na - hi, lo - na});
      if (out0 > tol && out1 <= out0 + tol) { continue; }
      mv_block = r;
      if (!frow_mixed[r]) {
        if (viol_int == nullptr) { return 1; }
        viol_int->push_back(r);
        mv_block = -1;
        continue;
      }
      const double need = na > hi + tol ? hi - na : lo - na;
      bool absorbed     = false;
      for (i_t p = fp_host.row_offsets[r]; p < fp_host.row_offsets[r + 1] && !absorbed; ++p) {
        const i_t j = fp_host.row_cols[p];
        if (is_int[j] || col_nrows[j] != 1 || fp_host.row_vals[p] == 0) { continue; }
        const double nx = (double)mv_x[j] + need / (double)fp_host.row_vals[p];
        if (nx >= h_vlb[j] - 1e-9 && nx <= h_vub[j] + 1e-9) {
          mv_cols.push_back({j, r, (double)fp_host.row_vals[p], nx, (double)mv_x[j]});
          absorbed = true;
        }
      }
      if (!absorbed) { return 2; }
      mv_block = -1;
    }
    for (i_t kk = fp_host.var_offsets[v]; kk < fp_host.var_offsets[v + 1]; ++kk) {
      mv_act[fp_host.var_rows[kk]] += (double)fp_host.var_vals[kk] * d;
    }
    for (const auto& m : mv_cols) {
      mv_act[m.row] += m.coef * (m.value - (double)mv_x[m.col]);
      mv_x[m.col] = (f_t)m.value;
    }
    return 0;
  };
  // takes back an accepted move of v by d whose absorbing columns were cols
  auto undo_move = [&](i_t v, double d, const std::vector<move_t>& cols) {
    for (i_t kk = fp_host.var_offsets[v]; kk < fp_host.var_offsets[v + 1]; ++kk) {
      mv_act[fp_host.var_rows[kk]] -= (double)fp_host.var_vals[kk] * d;
    }
    for (auto it = cols.rbegin(); it != cols.rend(); ++it) {
      mv_act[it->row] -= it->coef * (it->value - it->old);
      mv_x[it->col] = (f_t)it->old;
    }
  };
  auto set_link_keep = [&]() {
    mv_act.assign(fp_host.n_rows(), 0);
    for (i_t r = 0; r < fp_host.n_rows(); ++r) {
      double a = 0;
      for (i_t p = fp_host.row_offsets[r]; p < fp_host.row_offsets[r + 1]; ++p) {
        a += (double)fp_host.row_vals[p] * (double)anchor_x[fp_host.row_cols[p]];
      }
      mv_act[r] = a;
    }
    mv_x        = anchor_x;
    n_keep_used = n_keep_rows = n_keep_int_rows = n_close = 0;
    for (auto v : h_int) {
      const i_t k = link_of[v];
      link_keep[v] = 0;
      if (k < 0) { continue; }
      if (usage(k, anchor_x) > open_tol(k) && (!vub[k].pair || anchor_x[v] > 0.5)) {
        link_keep[v] = 1;
        n_keep_used++;
        continue;
      }
      if (!(anchor_x[v] > 0.5)) { continue; }
      const int m = try_move(v, -(double)anchor_x[v], vub[k].row, nullptr);
      if (m != 0) {
        link_keep[v] = 1;
        n_keep_int_rows += m == 1;
        n_keep_rows += m == 2;
        continue;
      }
      n_close++;
    }
  };
  // pattern of the current point, then fix-and-propagate (most used first), fixed LP, push,
  // repair, publish. mode 0: the slope-scaling point, the linked binaries with y = usage / scale
  // >= thr opened; mode 1: the used-link pattern of the current (published) point, every link
  // carrying flow open and every other link closed; mode 2: nb_pref / nb_key. The other integers
  // are rounded and the binaries of forced_open opened. Returns 0 evaluated, 1 duplicate,
  // 2 conflict, 3 no time.
  auto eval_pattern = [&](f_t thr, int mode) -> int {
    if (stop() || remaining() < 0.1 * budget) { return 3; }
    // the point of the pattern: the slope-scaling point (mode 0) or the anchored point
    const std::vector<f_t>& px = mode == 0 ? cur_x : anchor_x;
    const std::vector<f_t>& py = mode == 0 ? cur_y : anchor_y;
    std::vector<f_t> key(n_vars, 0), pref(n_vars, 0);
    i_t n_kept     = 0;  // linked binaries opened by the hysteresis only
    i_t n_pair_sum = 0;  // pair binaries opened by the pair's value sum
    if (mode == 1) {
      set_link_keep();
      CUOPT_LOG_INFO(
        "Slope scaling used-link pattern: used=%d kept_by_rows=%d kept_by_int_rows=%d closed=%d",
        n_keep_used,
        n_keep_rows,
        n_keep_int_rows,
        n_close);
    }
    for (auto v : h_int) {
      const i_t k = link_of[v];
      if (has_val(cert_val[v])) {
        key[v]  = 3;
        pref[v] = cert_val[v];
      } else if (forced_open[v]) {
        // fixed after the pattern's own open links (keys above 1), so that a forced binary that
        // conflicts with them is flipped instead of constraining them
        key[v]  = 1;
        pref[v] = 1;
      } else if (mode == 2) {
        key[v]  = nb_key[v];
        pref[v] = nb_pref[v];
      } else if (k >= 0 && mode == 1) {
        // used link (flow on its row, and for a pair binary also open at the point), or kept
        // open for another row of the point
        const bool o = link_keep[v];
        key[v]       = o ? 2 : 0;
        pref[v]      = o ? f_t(1) : f_t(0);
      } else if (k >= 0) {
        // a pair binary is opened by its own LP value (the LP chooses between the two binaries);
        // a binary open in the last published point stays open down to keep_frac * thr
        const f_t yk = vub[k].pair ? std::max<f_t>(cur_x[v], 0) : std::max<f_t>(u[k], 0) / scale[k];
        auto opens   = [&](i_t b, f_t y) {
          return y >= thr || (anchor_open[b] && y >= keep_frac * thr);
        };
        bool o = is_open[k] && opens(v, yk);
        n_kept += o && yk < thr;
        // a pair row whose flow the LP splits so that neither binary opens by itself, while the
        // two values sum to thr or more: the binary with the larger value opens
        const i_t p = pair_partner[k];
        if (!o && is_open[k] && p >= 0) {
          const i_t w  = vub[p].var;
          const f_t yw = std::max<f_t>(cur_x[w], 0);
          if (!opens(w, yw) && yk + yw >= thr && (yk > yw || (yk == yw && v < w))) {
            o = true;
            n_pair_sum++;
          }
        }
        key[v]  = o ? 1 + yk : 0;
        pref[v] = o ? f_t(1) : f_t(0);
      } else {
        const f_t x = std::clamp(px[v], h_vlb[v], h_vub[v]);
        key[v]      = x;
        pref[v]     = std::clamp<f_t>(std::floor(x + 0.5), h_vlb[v], h_vub[v]);
      }
    }
    // preference repair on the binary-only rows: a row the preferred values violate (a
    // cover "one of these must open" with all closed, a packing row with two open) is otherwise
    // settled by fix-and-propagate in fixing order, often too late (both values of its last
    // binary conflict). Each violated row flips the binary of the row whose flip lowers the
    // total violation of the binary-only rows most (the one with the LP value nearest its new
    // value on ties); binaries flipped open are fixed with the pattern's opens. In mode 1 the
    // used-link pattern closes the incumbent's open binaries without flow that these rows need;
    // the LP-nearest tie-break then reopens the incumbent's own binaries.
    // mode 1: linked binaries open in the published point (px) that the used-link preference
    // closes (before the preference repair)
    i_t n_pub_open = 0, n_pub_pref_closed0 = 0;
    if (mode == 1) {
      for (auto v : h_int) {
        if (link_of[v] < 0 || !(px[v] > 0.5)) { continue; }
        n_pub_open++;
        n_pub_pref_closed0 += pref[v] < 0.5;
      }
    }
    i_t n_bviol0 = 0, n_bviol1 = 0, n_bflips = 0;
    if (!brows.empty()) {
      const size_t nb = brows.size();
      std::vector<double> bact(nb, 0);
      auto bviol      = [&](size_t q, double a) {
        const i_t r = brows[q];
        double v    = 0;
        if (std::isfinite(h_cub[r]) && a > h_cub[r] + 1e-9) { v += a - h_cub[r]; }
        if (std::isfinite(h_clb[r]) && a < h_clb[r] - 1e-9) { v += h_clb[r] - a; }
        return v;
      };
      std::vector<size_t> bqueue;
      std::vector<char> bqueued(nb, 0);
      for (size_t q = 0; q < nb; ++q) {
        const i_t r = brows[q];
        double a    = 0;
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          a += (double)h_vals[p] * (double)pref[h_cols[p]];
        }
        bact[q] = a;
        if (bviol(q, a) > 0) {
          bqueue.push_back(q);
          bqueued[q] = 1;
          n_bviol0++;
        }
      }
      const i_t max_flips = 4 * n_bviol0 + 100;
      for (size_t head = 0; head < bqueue.size() && n_bflips < max_flips; ++head) {
        const size_t q = bqueue[head];
        bqueued[q]     = 0;
        const double v0 = bviol(q, bact[q]);
        if (v0 <= 0) { continue; }
        const i_t r       = brows[q];
        i_t best          = -1;
        double best_score = 0, best_dist = 0;
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          const i_t j = h_cols[p];
          if (forced_open[j] || has_val(cert_val[j]) || (mode == 2 && nb_key[j] >= 3)) {
            continue;
          }
          const double dj = pref[j] > 0.5 ? -1. : 1.;
          if (bviol(q, bact[q] + h_vals[p] * dj) >= v0) { continue; }
          double score = 0;
          for (i_t t = bvar_off[j]; t < bvar_off[j + 1]; ++t) {
            const i_t qq = bvar_row[t];
            score += bviol(qq, bact[qq] + bvar_val[t] * dj) - bviol(qq, bact[qq]);
          }
          if (score >= -1e-9) { continue; }
          const double dist =
            std::abs(std::clamp<double>(px[j], 0., 1.) - (pref[j] > 0.5 ? 0. : 1.));
          if (best < 0 || score < best_score - 1e-9 ||
              (score < best_score + 1e-9 && dist < best_dist)) {
            best       = j;
            best_score = score;
            best_dist  = dist;
          }
        }
        if (best < 0) { continue; }
        const double dj = pref[best] > 0.5 ? -1. : 1.;
        pref[best]      = pref[best] > 0.5 ? f_t(0) : f_t(1);
        key[best]       = pref[best] > 0.5 ? 1 + std::clamp<f_t>(px[best], 0, 1) : f_t(0);
        n_bflips++;
        for (i_t t = bvar_off[best]; t < bvar_off[best + 1]; ++t) {
          const i_t qq = bvar_row[t];
          bact[qq] += bvar_val[t] * dj;
          if (!bqueued[qq] && bviol(qq, bact[qq]) > 0) {
            bqueue.push_back(qq);
            bqueued[qq] = 1;
          }
        }
      }
      for (size_t q = 0; q < nb; ++q) {
        n_bviol1 += bviol(q, bact[q]) > 0;
      }
    }
    const double t_fp = timer.elapsed_time();
    i_t n_flipped = 0, n_implied = 0, n_conflicts = 0;
    // conflict promotion: a variable whose two values both conflicted is fixed first, at the
    // other end of its domain, in the next attempt (a closed binary of an infeasible cluster is
    // opened before its neighbours are fixed). The learned promotions of earlier patterns come
    // first (a binary moved by a flip or certificate repair keeps its value); one that conflicts
    // in this pattern is dropped for this pattern. Round patterns (mode 0) only: the
    // neighbourhood patterns keep the anchored point's design, which the promotions learned on
    // round patterns contradict (carried into them, half of the flip patterns conflicted).
    // A pattern that still conflicts teaches nothing: its promotions are not learned. One that
    // carried learned promotions is retried once without them (promotions that served earlier
    // patterns can make a later one infeasible at every attempt, and once learned they were
    // carried into every later pattern); when the retry succeeds, its promotions replace the
    // learned ones.
    std::vector<i_t> promoted;
    std::vector<char> is_promoted(n_vars, 0), carried(n_vars, 0);
    std::vector<f_t> pref_try;
    size_t n_carried        = 0;
    size_t n_carried0       = 0;  // learned promotions carried into the attempt
    auto fix_with_promotion = [&](bool with_learned) {
      std::vector<i_t> ord0(h_int.begin(), h_int.end());
      std::stable_sort(
        ord0.begin(), ord0.end(), [&](i_t a, i_t b) { return key[a] > key[b]; });
      promoted.clear();
      std::fill(is_promoted.begin(), is_promoted.end(), 0);
      std::fill(carried.begin(), carried.end(), 0);
      pref_try  = pref;
      n_carried = 0;
      for (const auto& [v, val] : learned) {
        if (mode != 0 || !with_learned) { break; }
        if ((forced_open[v] && val < pref[v]) || (has_val(cert_val[v]) && val != cert_val[v])) {
          continue;
        }
        promoted.push_back(v);
        is_promoted[v] = 1;
        carried[v]     = 1;
        pref_try[v]    = val;
        n_carried++;
      }
      n_carried0                 = n_carried;
      constexpr int max_attempts = 16;
      for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::vector<i_t> ord(promoted);
        for (auto v : ord0) {
          if (!is_promoted[v]) { ord.push_back(v); }
        }
        n_conflicts  = fp_host.fix_and_propagate(ord, pref_try, n_flipped, n_implied);
        const i_t cv = fp_host.last_conflict_var;
        if (n_conflicts > 0 && cv >= 0) {
          const i_t ck = link_of[cv];
          CUOPT_LOG_INFO(
            "Slope scaling conflict attempt %d: var=%d kind=%s pref=%g lp=%g key=%g usage=%g "
            "scale=%g",
            attempt,
            cv,
            ck < 0 ? "other" : (vub[ck].pair ? "pair" : "single"),
            (double)pref[cv],
            (double)cur_x[cv],
            (double)key[cv],
            ck < 0 ? 0. : (double)u[ck],
            ck < 0 ? 0. : (double)scale[ck]);
        }
        if (n_conflicts == 0 || cv < 0) { break; }
        if (is_promoted[cv]) {
          if (!carried[cv]) { break; }
          carried[cv]     = 0;
          is_promoted[cv] = 0;
          pref_try[cv]    = pref[cv];
          promoted.erase(std::find(promoted.begin(), promoted.end(), cv));
          n_carried--;
          continue;
        }
        promoted.push_back(cv);
        is_promoted[cv] = 1;
        pref_try[cv]    = pref_try[cv] > h_vlb[cv] ? h_vlb[cv] : h_vub[cv];
      }
    };
    fix_with_promotion(true);
    if (mode == 2 && n_conflicts > 0) {
      // flips that conflict when fixed first: fixed again after the used links of the anchored
      // point, so that propagation turns down the flips its kept links exclude
      for (auto v : h_int) {
        if (key[v] > 3 && !forced_open[v]) { key[v] = 1 + f_t(0.99) * (key[v] - 3); }
      }
      CUOPT_LOG_INFO("Slope scaling flip pattern conflicts: flips fixed after the used links");
      fix_with_promotion(true);
    }
    bool without_learned = false;
    if (mode == 0 && n_conflicts > 0 && n_carried0 > 0) {
      const size_t n_learned_carried = n_carried0;
      without_learned                = true;
      fix_with_promotion(false);
      CUOPT_LOG_INFO(
        "Slope scaling pattern retried without the %zu learned promotions: conflicts=%d "
        "learned_dropped=%d",
        n_learned_carried,
        n_conflicts,
        (int)(n_conflicts == 0));
    }
    if (mode == 0 && n_conflicts == 0) {
      if (without_learned) {
        for (const auto& lv : learned) {
          is_learned[lv.first] = 0;
        }
        learned.clear();
      }
      for (auto v : promoted) {
        if (!is_learned[v]) {
          is_learned[v] = 1;
          learned.push_back({v, pref_try[v]});
        }
      }
    }
    std::vector<f_t> pat(h_int.size());
    i_t n_ones = 0, n_pref_ones = 0;
    for (size_t qq = 0; qq < h_int.size(); ++qq) {
      pat[qq] = fp_host.col_lb[h_int[qq]];
      n_ones += pat[qq] > 0.5;
      n_pref_ones += pref[h_int[qq]] > 0.5;
    }
    for (size_t q = 0; q < rays.size() && n_conflicts == 0; ++q) {
      auto& ray = rays[q];
      double Dp = ray.D0, mag = std::abs(ray.D0);
      for (const auto& [j, z] : ray.z) {
        const double t = z * (double)fp_host.col_lb[j];
        Dp += t;
        mag += std::abs(t);
      }
      if (!(Dp > 1e-6 * mag)) { continue; }
      i_t n_more  = 0;
      double canc = 0;
      ray_moves(ray.z, [&](i_t j) { return (double)fp_host.col_lb[j]; }, Dp, n_more, canc);
      n_cert += n_more;
      ray.hits++;
      CUOPT_LOG_INFO(
        "Slope scaling certificate check: mode=%d thr=%g ray=%zu/%zu hits=%d proves the pattern "
        "infeasible (ray_objective/scale=%g) moved=%d cancelled/D=%g elapsed=%.2f",
        mode,
        (double)thr,
        q,
        rays.size(),
        ray.hits,
        Dp / mag,
        (int)n_more,
        canc / Dp,
        timer.elapsed_time());
      if (n_more > 0) { return 4; }
    }
    if (mode != 0) {
      // distance of the pattern from the anchored point, and (mode 2) the flips it kept
      i_t n_up = 0, n_down = 0, n_flips = 0, n_flips_kept = 0;
      for (size_t qq = 0; qq < h_int.size(); ++qq) {
        const i_t v   = h_int[qq];
        const f_t x0  = std::floor(px[v] + 0.5);
        n_up += pat[qq] > x0 + 0.5;
        n_down += pat[qq] < x0 - 0.5;
        if (mode == 2 && nb_key[v] >= 3) {
          n_flips++;
          n_flips_kept += std::abs(pat[qq] - nb_pref[v]) < 0.5;
        }
      }
      CUOPT_LOG_INFO("Slope scaling pattern moves: up=%d down=%d flips_kept=%d/%d",
                     n_up,
                     n_down,
                     n_flips_kept,
                     n_flips);
    }
    bool dup = false;
    for (auto& s : seen) {
      if (s == pat) { dup = true; }
    }
    CUOPT_LOG_INFO(
      "Slope scaling pattern mode=%d thr=%g forced=%d cert=%d: fix-and-propagate time=%.3fs "
      "conflicts=%d promoted=%zu carried=%zu flipped=%d implied_off_pref=%d pref_ones=%d ones=%d "
      "dup=%d elapsed=%.2f",
      mode,
      (double)thr,
      n_forced,
      n_cert,
      timer.elapsed_time() - t_fp,
      n_conflicts,
      promoted.size() - n_carried,
      n_carried,
      n_flipped,
      n_implied,
      n_pref_ones,
      n_ones,
      (int)dup,
      timer.elapsed_time());
    CUOPT_LOG_INFO(
      "Slope scaling pattern binary rows: mode=%d rows=%zu violated_by_preference=%d flips=%d "
      "still_violated=%d",
      mode,
      brows.size(),
      n_bviol0,
      n_bflips,
      n_bviol1);
    if (mode == 1) {
      // the published design against the used-link pattern: published-open linked binaries
      // closed by the preference (before / after the preference repair) and by the final
      // pattern, and published-closed ones the final pattern opens
      i_t n_pref_closed1 = 0, n_pat_closed = 0, n_pat_opened = 0;
      for (size_t qq = 0; qq < h_int.size(); ++qq) {
        const i_t v = h_int[qq];
        if (link_of[v] < 0) { continue; }
        const bool po = px[v] > 0.5;
        n_pref_closed1 += po && pref[v] < 0.5;
        n_pat_closed += po && pat[qq] < 0.5;
        n_pat_opened += !po && pat[qq] > 0.5;
      }
      CUOPT_LOG_INFO(
        "Slope scaling used-link pattern vs published: published_open=%d pref_closed=%d "
        "pref_closed_after_repair=%d pattern_closed=%d pattern_opened_other=%d",
        n_pub_open,
        n_pub_pref_closed0,
        n_pref_closed1,
        n_pat_closed,
        n_pat_opened);
    }
    if (mode == 0) {
      // link rows carrying flow at the point that the pattern closes: by the threshold (for a
      // pair row: with the two binaries' values summing to the threshold or more) or by
      // fix-and-propagate (a binary the pattern opened)
      i_t n_used = 0, n_closed_thr = 0, n_closed_pair_sum = 0, n_closed_fp = 0;
      double flow_used = 0, flow_closed = 0;
      for (size_t k = 0; k < L; ++k) {
        const i_t p = pair_partner[k];
        if ((p >= 0 && p < (i_t)k) || !(u[k] > open_tol(k))) { continue; }
        n_used++;
        flow_used += u[k];
        const i_t v = vub[k].var, w = p >= 0 ? vub[p].var : -1;
        if (fp_host.col_lb[v] > 0.5 || (w >= 0 && fp_host.col_lb[w] > 0.5)) { continue; }
        flow_closed += u[k];
        if (pref[v] > 0.5 || (w >= 0 && pref[w] > 0.5)) {
          n_closed_fp++;
        } else {
          n_closed_thr++;
          n_closed_pair_sum += w >= 0 && std::max<f_t>(cur_x[v], 0) + std::max<f_t>(cur_x[w], 0) >= thr;
        }
      }
      CUOPT_LOG_INFO(
        "Slope scaling pattern closes used link rows: used=%d closed_by_threshold=%d "
        "(pair_sum_above=%d) closed_by_propagation=%d closed_flow_share=%.3f kept_by_anchor=%d "
        "pair_sum_opened=%d",
        n_used,
        n_closed_thr,
        n_closed_pair_sum,
        n_closed_fp,
        flow_used > 0 ? flow_closed / flow_used : 0.,
        n_kept,
        n_pair_sum);
      if (published) {
        // the pattern against the last published point: its open link rows the pattern closes
        // (and their share of its flow), and link rows it opens that the published point closes
        i_t n_pub_rows = 0, n_pub_closed = 0, n_new_rows = 0;
        double pub_flow = 0, pub_flow_closed = 0;
        for (size_t k = 0; k < L; ++k) {
          const i_t p = pair_partner[k];
          if (p >= 0 && p < (i_t)k) { continue; }
          const i_t v = vub[k].var, w = p >= 0 ? vub[p].var : -1;
          const bool pub_open = anchor_open[v] || (w >= 0 && anchor_open[w]);
          const bool pat_open = fp_host.col_lb[v] > 0.5 || (w >= 0 && fp_host.col_lb[w] > 0.5);
          if (pub_open) {
            n_pub_rows++;
            pub_flow += std::max<f_t>(anchor_u[k], 0);
            if (!pat_open) {
              n_pub_closed++;
              pub_flow_closed += std::max<f_t>(anchor_u[k], 0);
            }
          } else if (pat_open) {
            n_new_rows++;
          }
        }
        CUOPT_LOG_INFO(
          "Slope scaling pattern vs published: published_open_rows=%d closed=%d "
          "closed_flow_share=%.3f opened_rows=%d",
          n_pub_rows,
          n_pub_closed,
          pub_flow > 0 ? pub_flow_closed / pub_flow : 0.,
          n_new_rows);
      }
      // mixed coupling rows the pattern's binaries overload at the point's flows (before the
      // fixed LP moves the flows): rounding a linked binary up takes more of the row
      i_t n_over = 0;
      double max_over = 0, sum_over = 0, sum_extra = 0;
      for (auto r : mixed_coupling_rows) {
        double act = 0, d = 0, mag = 0;
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          const i_t c = h_cols[p];
          act += (double)h_vals[p] * (double)cur_x[c];
          mag = std::max(mag, std::abs((double)h_vals[p] * (double)cur_x[c]));
          if (is_vub_bin[c]) { d += (double)h_vals[p] * ((double)fp_host.col_lb[c] - (double)cur_x[c]); }
        }
        const bool le     = std::isfinite(h_cub[r]);
        const double sgn  = le ? 1. : -1.;
        const double over = sgn * (act + d) - std::max(sgn * (le ? h_cub[r] : h_clb[r]), sgn * act);
        sum_extra += sgn * d;
        if (over > 1e-6 * std::max(1., mag)) {
          n_over++;
          max_over = std::max(max_over, over);
          sum_over += over;
        }
      }
      CUOPT_LOG_INFO(
        "Slope scaling pattern coupling rows: mixed=%zu overloaded=%d max_over=%g sum_over=%g "
        "binary_extra_use=%g",
        mixed_coupling_rows.size(),
        n_over,
        max_over,
        sum_over,
        sum_extra);
    }
    if (dup) { return 1; }
    if (n_conflicts > 0) { return 2; }
    seen.push_back(pat);
    std::vector<f_t> assign = px;
    for (i_t j = 0; j < n_vars; ++j) {
      assign[j] = std::clamp(assign[j], h_vlb[j], h_vub[j]);
    }
    for (size_t qq = 0; qq < h_int.size(); ++qq) {
      assign[h_int[qq]] = pat[qq];
    }
    const f_t rem = remaining();
    // the retry of a flip-repaired point gets less fixed-LP time, and no push or repair when its
    // fixed LP ends at more than twice the first attempt's residual (a failed retry cost ~130 s);
    // a neighbourhood pattern of the anchored point (warm-started from it) settles its fixed-LP
    // objective within ~15 s, the residual left after that is the pushes' work; a safe step's
    // (its moved point feasible, the fixed LP below the incumbent) gets up to 45 s, its first
    // step after a mode-0 publish (~1000 flips) stopped unconverged at 20 s 3 runs out of 4
    const bool retry       = mode == 0 && point_lps > 0;
    const f_t rank_cap     = safe_warm   ? f_t(45)
                             : mode != 0 ? f_t(20)
                                         : (retry ? f_t(40) : f_t(60));
    const f_t rank_tlimit  = std::min<f_t>(rank_cap, 0.45 * rem);
    const f_t repair_limit = std::min<f_t>(40., 0.35 * rem);
    // no repair for patterns whose fixed LP is not below the incumbent
    const f_t repair_below = population.is_feasible()
                               ? population.best_feasible().get_user_objective()
                               : std::numeric_limits<f_t>::infinity();
    // a safe step's moved point is feasible for its fixed LP
    const std::vector<f_t>& wx = safe_warm && mv_x.size() == (size_t)n_vars ? mv_x : px;
    // round patterns are repaired (and their LP not cut off) up to a near miss
    const f_t repair_near = mode == 0 && std::isfinite(repair_below)
                              ? repair_below + near_frac * std::abs(repair_below)
                              : repair_below;
    auto ev               = eval_fixed_pattern(assign,
                                 1e-4,
                                 rank_tlimit,
                                 &wx,
                                 &py,
                                 true,
                                 repair_limit,
                                 "slope_scaling",
                                 repair_near,
                                 11.,
                                 retry ? 2 * first_try_res : std::numeric_limits<f_t>::infinity(),
                                 repair_near);
    if (mode == 0 && !ev.published && ev.repaired_assignment.size() == (size_t)n_vars &&
        ev.dual.size() == (size_t)n_cstrs && ev.repaired_obj < repair_near &&
        ev.repaired_obj < near_obj) {
      near_x   = ev.repaired_assignment;
      near_y   = ev.dual;
      near_obj = ev.repaired_obj;
      CUOPT_LOG_INFO("Slope scaling near miss: repaired=%g incumbent=%g (+%.3f%%)",
                     ev.repaired_obj,
                     repair_below,
                     100. * (ev.repaired_obj - repair_below) / std::abs(repair_below));
    }
    if (mode == 0 && !retry) { first_try_res = ev.l2_primal_res; }
    if (mode == 0) { point_lps++; }
    if (ev.aborted) {
      CUOPT_LOG_INFO(
        "Slope scaling retry aborted: l2_primal_res=%g above twice the first attempt's %g time=%.2fs",
        ev.l2_primal_res,
        first_try_res,
        ev.time);
    }
    n_evals++;
    eval_times.push_back(ev.time);
    published     = published || ev.published;
    last_eval_obj = ev.published ? ev.repaired_obj : std::numeric_limits<f_t>::infinity();
    // flip repair: the fixed-LP point of a pattern below the incumbent that could not be repaired
    // violates rows; in each (largest violation first) the closed link rows of the row are
    // opened, and also those whose continuous columns appear in the row (a violated flow row is
    // relieved by opening the links that carry its flows); one binary per pair row, the one with
    // the larger value at the slope-scaling point. The opened binaries (forced_open) are used for
    // one more evaluation of the same point (mode 0 only, the retry is not repaired again). The
    // violated rows are taken at the pushed point when there is one (the residual of a pattern
    // the fixed LP cannot serve stays on the rows it cannot serve).
    i_t n_new_forced = 0;
    if (mode == 0 && n_forced == 0 && n_cert == 0 && !ev.published && ev.usable &&
        !ev.infeasible && !ev.cutoff && ev.lp_obj < repair_below &&
        ev.lp_assignment.size() == (size_t)n_vars) {
      const auto& x =
        ev.pushed_assignment.size() == (size_t)n_vars ? ev.pushed_assignment : ev.lp_assignment;
      std::vector<std::pair<double, i_t>> viol_rows;
      for (i_t r = 0; r < n_cstrs; ++r) {
        double act = 0, mag = 0;
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          act += (double)h_vals[p] * (double)x[h_cols[p]];
          mag = std::max(mag, std::abs((double)h_vals[p] * (double)x[h_cols[p]]));
        }
        const double tol = 10. * abs_tol * std::max(1., mag);
        double v         = 0;
        if (std::isfinite(h_clb[r]) && act < h_clb[r] - tol) { v = h_clb[r] - act; }
        if (std::isfinite(h_cub[r]) && act > h_cub[r] + tol) { v = act - h_cub[r]; }
        if (v > 0) { viol_rows.push_back({v, r}); }
      }
      std::sort(viol_rows.begin(), viol_rows.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
      });
      constexpr i_t max_new_forced = 500;
      i_t n_direct = 0, n_via_flow = 0;
      // a link row is open in the pattern if one of its binaries is
      auto row_open = [&](i_t k) {
        if (x[vub[k].var] >= 0.5) { return true; }
        const i_t p = pair_partner[k];
        return p >= 0 && x[vub[p].var] >= 0.5;
      };
      auto open_link = [&](i_t k, i_t& counter) {
        if (n_new_forced >= max_new_forced || row_open(k)) { return; }
        i_t v       = vub[k].var;
        const i_t p = pair_partner[k];
        if (p >= 0) {
          const i_t w = vub[p].var;
          if (forced_open[w]) { return; }
          if (cur_x[w] > cur_x[v] || (cur_x[w] == cur_x[v] && vub[p].cap > vub[k].cap)) { v = w; }
        }
        if (forced_open[v]) { return; }
        forced_open[v] = 1;
        n_new_forced++;
        counter++;
      };
      // diagnostics: the closed link rows next to the largest violations, the closed link rows
      // the slope-scaling point uses, and the closed link rows with the largest fixed-LP duals
      auto row_once = [&](i_t k) { return pair_partner[k] < 0 || k < pair_partner[k]; };
      auto closed_used = [&](i_t k) { return !row_open(k) && u[k] > open_tol(k); };
      auto closed_pref = [&](i_t k) {
        const i_t p = pair_partner[k];
        return !row_open(k) && (pref[vub[k].var] > 0.5 || (p >= 0 && pref[vub[p].var] > 0.5));
      };
      std::vector<char> is_cand(L, 0);
      for (size_t qv = 0; qv < viol_rows.size(); ++qv) {
        const i_t r = viol_rows[qv].second;
        i_t n_c = 0, n_cl = 0, n_cu = 0, n_cp = 0, n_int_r = 0;
        double act = 0;
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          const i_t c = h_cols[p];
          act += (double)h_vals[p] * (double)x[c];
          n_int_r += is_int[c];
          auto visit = [&](i_t k) {
            if (is_cand[k] || !row_once(k)) { return; }
            is_cand[k] = 1;
            if (qv >= 3) { return; }
            n_c++;
            n_cl += !row_open(k);
            n_cu += closed_used(k);
            n_cp += closed_pref(k);
          };
          if (link_of[c] >= 0) {
            const i_t k = link_of[c];
            visit(row_once(k) ? k : pair_partner[k]);
          } else if (!is_int[c]) {
            for (i_t q = col_link_off[c]; q < col_link_off[c + 1]; ++q) {
              const i_t k = col_link[q];
              visit(row_once(k) ? k : pair_partner[k]);
            }
          }
        }
        if (qv < 3) {
          CUOPT_LOG_INFO(
            "Slope scaling flip repair row %d: lo=%g hi=%g act=%g viol=%g nnz=%d ints=%d "
            "link_rows=%d closed=%d closed_used_at_point=%d closed_pref_open=%d dual=%g",
            r,
            (double)h_clb[r],
            (double)h_cub[r],
            act,
            viol_rows[qv].first,
            h_offsets[r + 1] - h_offsets[r],
            n_int_r,
            n_c,
            n_cl,
            n_cu,
            n_cp,
            ev.dual.size() == (size_t)n_cstrs ? (double)ev.dual[r] : 0.);
        }
      }
      {
        i_t n_cu = 0, n_cp = 0, n_cl = 0;
        std::vector<std::pair<double, i_t>> by_dual;
        for (size_t k = 0; k < L; ++k) {
          if (!row_once(k) || row_open(k)) { continue; }
          n_cl++;
          n_cu += closed_used(k);
          n_cp += closed_pref(k);
          if (ev.dual.size() == (size_t)n_cstrs) {
            const i_t p   = pair_partner[k];
            const f_t cap = p >= 0 ? std::max(vub[k].cap, vub[p].cap) : vub[k].cap;
            by_dual.push_back({std::abs((double)ev.dual[vub[k].row]) * cap, (i_t)k});
          }
        }
        const size_t top = std::min<size_t>(20, by_dual.size());
        std::partial_sort(by_dual.begin(),
                          by_dual.begin() + top,
                          by_dual.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        i_t top_cand = 0, top_used = 0;
        for (size_t t = 0; t < top; ++t) {
          top_cand += is_cand[by_dual[t].second];
          top_used += closed_used(by_dual[t].second);
        }
        CUOPT_LOG_INFO(
          "Slope scaling flip repair links: closed_rows=%d closed_used_at_point=%d "
          "closed_pref_open=%d top20_dual(first=%g last=%g) in_candidates=%d used_at_point=%d",
          n_cl,
          n_cu,
          n_cp,
          top > 0 ? by_dual[0].first : 0.,
          top > 0 ? by_dual[top - 1].first : 0.,
          top_cand,
          top_used);
      }
      for (const auto& [v, r] : viol_rows) {
        if (n_new_forced >= max_new_forced) { break; }
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          const i_t c = h_cols[p];
          if (link_of[c] >= 0) {
            open_link(link_of[c], n_direct);
          } else if (!is_int[c]) {
            for (i_t q = col_link_off[c]; q < col_link_off[c + 1]; ++q) {
              open_link(col_link[q], n_via_flow);
            }
          }
        }
      }
      n_forced += n_new_forced;
      CUOPT_LOG_INFO(
        "Slope scaling flip repair: violated_rows=%zu max_violation=%g point=%s opened direct=%d "
        "via_flow=%d "
        "elapsed=%.2f",
        viol_rows.size(),
        viol_rows.empty() ? 0. : viol_rows[0].first,
        ev.pushed_assignment.size() == (size_t)n_vars ? "pushed" : "lp",
        n_direct,
        n_via_flow,
        timer.elapsed_time());
    }
    // certificate repair of an infeasible fixed LP: the dual iterate y of the stopped
    // PDLP run diverges along a Farkas ray. With z = -A^T y its ray objective D (row terms y_i *
    // the bound of the side y_i holds, column terms z_j * the bound of the side z_j holds, a
    // fixed integer at its value) is positive. Moving a fixed integer j by delta changes D by
    // z_j * delta, so the integers whose move cancels the most of D (closing a binary that
    // consumes a tight capacity row, opening one that relieves it) are moved (cert_val) for the
    // retries of the same point, until the cancelled part covers twice D. A fixed LP stopped on
    // its Lagrangian bound may be an infeasible one whose dual grows along such a ray (the bound
    // exceeds the incumbent before the divergence verdict): its dual is checked the same way, and
    // only a positive D (a valid Farkas proof, no free column against it) moves integers
    i_t n_new_cert = 0;
    if ((ev.infeasible || ev.cutoff == 1) && !ev.published && ev.dual.size() == (size_t)n_cstrs) {
      const auto& y = ev.dual;
      double ymax   = 0;
      std::vector<std::pair<double, i_t>> by_y;
      for (i_t r = 0; r < n_cstrs; ++r) {
        if (!std::isfinite((double)y[r])) { continue; }
        ymax = std::max(ymax, std::abs((double)y[r]));
        if (y[r] != 0) { by_y.push_back({std::abs((double)y[r]), r}); }
      }
      // the rows carrying the ray (largest |y|): the deficit rows of the pattern
      std::string rows;
      {
        const size_t top = std::min<size_t>(4, by_y.size());
        std::partial_sort(by_y.begin(),
                          by_y.begin() + top,
                          by_y.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t t = 0; t < top; ++t) {
          char buf[64];
          snprintf(buf, sizeof(buf), " %d:%.3g", (int)by_y[t].second, by_y[t].first / ymax);
          rows += buf;
        }
      }
      std::vector<double> aty(n_vars, 0.);
      double D = 0, D_mag = 0;
      i_t n_bad_rows = 0, n_bad_cols = 0;
      for (i_t r = 0; r < n_cstrs && ymax > 0; ++r) {
        const double yr = y[r];
        if (!(std::abs(yr) > 1e-12 * ymax)) { continue; }
        const double b = yr > 0 ? h_clb[r] : h_cub[r];
        if (!std::isfinite(b)) {
          n_bad_rows++;
          continue;
        }
        D += yr * b;
        D_mag += std::abs(yr * b);
        for (i_t p = h_offsets[r]; p < h_offsets[r + 1]; ++p) {
          aty[h_cols[p]] += (double)h_vals[p] * yr;
        }
      }
      double zmax = 0;
      for (i_t j = 0; j < n_vars; ++j) {
        zmax = std::max(zmax, std::abs(aty[j]));
      }
      std::vector<std::pair<i_t, double>> z_int;
      double D_int = 0;
      for (i_t j = 0; j < n_vars && zmax > 0; ++j) {
        const double z = -aty[j];
        if (std::abs(z) <= 1e-12 * zmax) { continue; }
        if (!is_int[j]) {
          const double b = z > 0 ? h_vlb[j] : h_vub[j];
          if (!std::isfinite(b)) {
            n_bad_cols++;
            continue;
          }
          D += z * b;
          D_mag += std::abs(z * b);
          continue;
        }
        z_int.push_back({j, z});
        D_int += z * (double)assign[j];
        D_mag += std::abs(z * (double)assign[j]);
      }
      D += D_int;
      const bool proof = D > 1e-6 * D_mag && n_bad_cols == 0 && !z_int.empty();
      double cancelled = 0;
      std::vector<std::pair<i_t, f_t>> taken;
      if (ev.infeasible || proof) {
        ev.infeasible = true;
        taken         = ray_moves(
          z_int, [&](i_t j) { return (double)assign[j]; }, D, n_new_cert, cancelled);
      }
      i_t n_opening = 0;
      std::string top;
      for (const auto& [j, t] : taken) {
        n_opening += t > assign[j];
        if (top.size() < 400) {
          const i_t k = link_of[j];
          char buf[160];
          snprintf(buf,
                   sizeof(buf),
                   " [var=%d %s %g->%g pref=%g point=%g]",
                   (int)j,
                   k < 0 ? "int" : (vub[k].pair ? "pair" : "single"),
                   (double)assign[j],
                   (double)t,
                   (double)pref[j],
                   (double)cur_x[j]);
          top += buf;
        }
      }
      n_cert += n_new_cert;
      const size_t n_z_int = z_int.size();
      if (proof) {
        if (rays.size() >= max_rays) { rays.erase(rays.begin()); }
        rays.push_back({std::move(z_int), D - D_int, 0});
      }
      CUOPT_LOG_INFO(
        "Slope scaling certificate: mode=%d thr=%g stop=%s rows(|y|/ymax):%s D=%g D/mag=%g "
        "ymax=%g bad_rows=%d bad_cols=%d int_terms=%zu moved=%d opening=%d cancelled/D=%g "
        "rays=%zu top:%s elapsed=%.2f",
        mode,
        (double)thr,
        ev.cutoff == 1 ? "bound" : "divergence",
        rows.c_str(),
        D,
        D_mag > 0 ? D / D_mag : 0.,
        ymax,
        (int)n_bad_rows,
        (int)n_bad_cols,
        n_z_int,
        (int)n_new_cert,
        (int)n_opening,
        D > 0 ? cancelled / D : 0.,
        rays.size(),
        top.c_str(),
        timer.elapsed_time());
    }
    if (ev.published && ev.repaired_assignment.size() == (size_t)n_vars &&
        ev.dual.size() == (size_t)n_cstrs) {
      cur_x      = ev.repaired_assignment;
      cur_y      = ev.dual;
      anchor_x   = cur_x;
      anchor_y   = cur_y;
      anchor_obj = ev.repaired_obj;
      for (size_t k = 0; k < L; ++k) {
        u[k]                    = usage(k, cur_x);
        is_open[k]              = u[k] > open_tol(k);
        prev_open[k]            = is_open[k];
        anchor_open[vub[k].var] = cur_x[vub[k].var] > 0.5;
        anchor_u[k]             = u[k];
      }
      CUOPT_LOG_INFO("Slope scaling anchored on the published point obj=%g elapsed=%.2f",
                     ev.repaired_obj,
                     timer.elapsed_time());
    }
    CUOPT_LOG_INFO(
      "Slope scaling eval %d mode=%d thr=%g: ones=%d lp_obj=%g optimal=%d infeasible=%d "
      "cutoff=%d iters=%d l2_primal_res=%g time=%.2fs (lp/push/repair %.1f/%.1f/%.1f) "
      "pushed_lp_obj=%g pushed_res=%g repaired=%g published=%d incumbent=%g elapsed=%.2f",
      n_evals,
      mode,
      (double)thr,
      n_ones,
      ev.lp_obj,
      (int)ev.optimal,
      (int)ev.infeasible,
      (int)ev.cutoff,
      ev.iters,
      ev.l2_primal_res,
      ev.time,
      ev.lp_time,
      ev.push_time,
      ev.repair_time,
      ev.pushed_lp_obj,
      ev.pushed_l2_primal_res,
      ev.repaired_obj,
      (int)ev.published,
      repair_below,
      timer.elapsed_time());
    // 4: flip- or certificate-repaired, worth evaluating the point again
    return n_new_forced > 0 || n_new_cert > 0 ? 4 : 0;
  };
  // the flip and certificate moves of a point (for its retries only) and its LP count
  auto clear_moves = [&]() {
    std::fill(forced_open.begin(), forced_open.end(), 0);
    n_forced = 0;
    std::fill(cert_val.begin(), cert_val.end(), no_val);
    n_cert    = 0;
    point_lps = 0;
  };
  // safe neighbourhood step of the anchored point: its used-link pattern (the flow-free links whose
  // closing keeps the point within its rows closed, see set_link_keep), plus the binaries whose
  // flip is estimated to gain by its reduced cost at the anchored point's fixed-LP duals (rows with
  // continuous columns), largest estimate first, each kept only if the anchored point, moved by
  // the flips kept so far, stays within every row of the binary (single-row continuous columns
  // absorbing what they can, e.g. a fixed cost). The anchored point then stays feasible for the
  // step's fixed LP, whose value is at most the incumbent's; fix-and-propagate has nothing to
  // settle. Binaries flipped by a step that did not gain enough are tabu until the next published
  // point. Returns eval_pattern's code, or 5 when the step would not move the point.
  constexpr i_t max_safe_flips = 1024;
  std::vector<char> safe_tabu(n_vars, 0);
  std::vector<i_t> last_safe;
  i_t n_tabu   = 0;
  f_t tabu_obj = std::numeric_limits<f_t>::infinity();  // the anchored point the tabu is for
  std::vector<f_t> rc_of(n_vars, 0);  // reduced costs of the binaries at the anchored point
  std::vector<char> moved(n_vars, 0);  // binaries the step moves
  auto safe_step = [&]() -> int {
    last_safe.clear();
    if (anchor_y.size() != (size_t)n_cstrs || anchor_x.size() != (size_t)n_vars ||
        !std::isfinite(anchor_obj)) {
      return 5;
    }
    if (anchor_obj != tabu_obj) {
      std::fill(safe_tabu.begin(), safe_tabu.end(), 0);
      n_tabu   = 0;
      tabu_obj = anchor_obj;
    }
    const f_t eps = f_t(1e-7) * std::max<f_t>(1, std::abs(anchor_obj));
    std::vector<std::pair<f_t, i_t>> cand;
    set_link_keep();
    for (auto v : h_int) {
      const i_t k = link_of[v];
      nb_pref[v]  = std::clamp<f_t>(std::floor(anchor_x[v] + 0.5), h_vlb[v], h_vub[v]);
      if (k >= 0) {
        nb_pref[v] = link_keep[v] ? f_t(1) : f_t(0);
        nb_key[v]  = link_keep[v] ? 2 : 0;
      } else {
        nb_key[v] = nb_pref[v];
      }
      if (h_vlb[v] != 0 || h_vub[v] != 1 || safe_tabu[v]) { continue; }
      double aty = 0;
      for (i_t kk = fp_host.var_offsets[v]; kk < fp_host.var_offsets[v + 1]; ++kk) {
        const i_t r = fp_host.var_rows[kk];
        if (frow_mixed[r]) { aty += (double)fp_host.var_vals[kk] * anchor_y[fp_host.row_orig[r]]; }
      }
      const f_t rc   = (f_t)((double)h_obj[v] - aty);
      rc_of[v]       = rc;
      const f_t gain = nb_pref[v] > 0.5 ? rc : -rc;
      if (gain > eps) { cand.push_back({gain, v}); }
    }
    std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    i_t n_rej_int = 0, n_rej_rows = 0, n_open = 0, n_pairs = 0, n_partners = 0;
    double gain_kept = 0;
    std::fill(moved.begin(), moved.end(), 0);
    // compound move: a move of v by d that only integer-only rows refuse (at most
    // max_compound_rows of them) is made anyway, and each row it takes out is put back by the move
    // of one of its other binaries (the one that gains the most by its reduced cost, among those
    // whose move puts the row back and keeps the point within its other rows), if all of them
    // together gain. On failure every move is taken back.
    constexpr size_t max_compound_rows = 3;
    struct comp_move_t {
      i_t var;
      double d;
      std::vector<move_t> cols;
    };
    std::vector<comp_move_t> comp;
    std::vector<i_t> comp_rows;
    std::vector<std::pair<f_t, i_t>> partners;
    auto row_in = [&](i_t r) {
      const double lo = fp_host.row_lb[r], hi = fp_host.row_ub[r], a = mv_act[r];
      const double tol =
        1e-6 * std::max({1., std::isfinite(lo) ? std::abs(lo) : 0., std::isfinite(hi) ? std::abs(hi) : 0.});
      return a <= hi + tol && a >= lo - tol;
    };
    auto undo_comp = [&]() {
      for (auto it = comp.rbegin(); it != comp.rend(); ++it) {
        undo_move(it->var, it->d, it->cols);
        moved[it->var] = 0;
      }
      comp.clear();
    };
    auto try_compound = [&](i_t v, double d, f_t g) -> bool {
      comp.clear();
      if (try_move(v, d, -1, &comp_rows) != 0) { return false; }
      comp.push_back({v, d, mv_cols});
      moved[v] = 1;
      if (comp_rows.size() > max_compound_rows) {
        undo_comp();
        return false;
      }
      double gsum = g;
      for (const i_t r : comp_rows) {
        if (row_in(r)) { continue; }
        partners.clear();
        const double lo = fp_host.row_lb[r], hi = fp_host.row_ub[r];
        const double tol =
          1e-6 * std::max({1., std::isfinite(lo) ? std::abs(lo) : 0., std::isfinite(hi) ? std::abs(hi) : 0.});
        for (i_t p = fp_host.row_offsets[r]; p < fp_host.row_offsets[r + 1]; ++p) {
          const i_t w = fp_host.row_cols[p];
          if (moved[w] || safe_tabu[w] || h_vlb[w] != 0 || h_vub[w] != 1) { continue; }
          const double dw = nb_pref[w] > 0.5 ? -1. : 1.;
          const double na = mv_act[r] + (double)fp_host.row_vals[p] * dw;
          if (na > hi + tol || na < lo - tol) { continue; }
          partners.push_back({-(f_t)dw * rc_of[w], w});
        }
        std::sort(partners.begin(), partners.end(), [](const auto& a, const auto& b) {
          return a.first > b.first || (a.first == b.first && a.second < b.second);
        });
        bool ok = false;
        for (const auto& [gw, w] : partners) {
          const double dw = nb_pref[w] > 0.5 ? -1. : 1.;
          if (try_move(w, dw, -1, nullptr) != 0) { continue; }
          comp.push_back({w, dw, mv_cols});
          moved[w] = 1;
          gsum += gw;
          ok = true;
          break;
        }
        if (!ok) {
          undo_comp();
          return false;
        }
      }
      for (const i_t r : comp_rows) {
        if (!row_in(r)) {
          undo_comp();
          return false;
        }
      }
      if (!(gsum > eps)) {
        undo_comp();
        return false;
      }
      return true;
    };
    // refused moves: [opening][int-only row, row with continuous columns]; the integer-only rows
    // that refused openings by their number of terms (2, 3, 4-10, more), and the most frequent
    i_t n_blk[2][2] = {}, blk_len[4] = {};
    std::unordered_map<i_t, i_t> blk_rows;
    // openings refused by rows with continuous columns: with a single-row continuous column (at
    // its bound) or without, and the most frequent rows
    i_t n_mix_single = 0;
    std::unordered_map<i_t, i_t> mix_rows;
    for (const auto& [g, v] : cand) {
      if ((i_t)last_safe.size() >= max_safe_flips) { break; }
      if (moved[v]) { continue; }
      const bool up = nb_pref[v] < 0.5;
      const int m   = try_move(v, up ? 1. : -1., -1, nullptr);
      if (m == 1 && mv_block >= 0) {
        const i_t r = mv_block;
        if (try_compound(v, up ? 1. : -1., g)) {
          for (const auto& c : comp) {
            last_safe.push_back(c.var);
            n_open += c.d > 0;
            gain_kept += c.var == v ? (double)g : -(double)c.d * (double)rc_of[c.var];
          }
          n_pairs++;
          n_partners += (i_t)comp.size() - 1;
          comp.clear();
          continue;
        }
        mv_block = r;
      }
      n_rej_int += m == 1;
      n_rej_rows += m == 2;
      if (m != 0) {
        n_blk[up][m - 1]++;
        if (m == 1 && up && mv_block >= 0) {
          const i_t len = fp_host.row_offsets[mv_block + 1] - fp_host.row_offsets[mv_block];
          blk_len[len <= 2 ? 0 : len == 3 ? 1 : len <= 10 ? 2 : 3]++;
          blk_rows[mv_block]++;
        }
        if (m == 2 && up && mv_block >= 0) {
          bool single = false;
          for (i_t p = fp_host.row_offsets[mv_block]; p < fp_host.row_offsets[mv_block + 1]; ++p) {
            const i_t j = fp_host.row_cols[p];
            single      = single || (!is_int[j] && col_nrows[j] == 1);
          }
          n_mix_single += single;
          mix_rows[mv_block]++;
        }
        continue;
      }
      moved[v] = 1;
      last_safe.push_back(v);
      n_open += up;
      gain_kept += g;
    }
    const size_t n_kept = last_safe.size();
    for (size_t q = 0; q < n_kept; ++q) {
      const i_t v = last_safe[q];
      nb_pref[v]  = 1 - nb_pref[v];
      nb_key[v]   = 3 + (f_t)(n_kept - q) / (f_t)n_kept;
    }
    CUOPT_LOG_INFO(
      "Slope scaling safe step: used=%d closed=%d kept_by_rows=%d kept_by_int_rows=%d "
      "candidates=%zu tabu=%d flipped=%zu (compound=%d partners=%d) opened=%d closed_by_flip=%d "
      "rejected_int_rows=%d rejected_rows=%d est_gain_top=%g est_gain_kept=%g elapsed=%.2f",
      n_keep_used,
      n_close,
      n_keep_rows,
      n_keep_int_rows,
      cand.size(),
      n_tabu,
      n_kept,
      n_pairs,
      n_partners,
      n_open,
      (int)n_kept - n_open,
      n_rej_int,
      n_rej_rows,
      cand.empty() ? 0. : (double)cand[0].first,
      gain_kept,
      timer.elapsed_time());
    {
      std::vector<std::pair<i_t, i_t>> top(blk_rows.begin(), blk_rows.end());
      const size_t nt = std::min<size_t>(3, top.size());
      std::partial_sort(top.begin(), top.begin() + nt, top.end(), [](const auto& a, const auto& b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
      });
      std::string tops;
      for (size_t t = 0; t < nt; ++t) {
        const i_t r = top[t].first;
        tops += " row" + std::to_string(fp_host.row_orig[r]) + "(terms=" +
                std::to_string(fp_host.row_offsets[r + 1] - fp_host.row_offsets[r]) +
                " lb=" + std::to_string((double)fp_host.row_lb[r]) +
                " ub=" + std::to_string((double)fp_host.row_ub[r]) +
                " act=" + std::to_string(mv_act[r]) + ")x" + std::to_string(top[t].second);
      }
      CUOPT_LOG_INFO(
        "Slope scaling safe step refusals: open int_rows=%d rows=%d close int_rows=%d rows=%d "
        "open int_row_terms(2/3/4-10/more)=%d/%d/%d/%d distinct_rows=%zu top:%s",
        n_blk[1][0],
        n_blk[1][1],
        n_blk[0][0],
        n_blk[0][1],
        blk_len[0],
        blk_len[1],
        blk_len[2],
        blk_len[3],
        blk_rows.size(),
        tops.c_str());
      std::vector<std::pair<i_t, i_t>> mtop(mix_rows.begin(), mix_rows.end());
      const size_t nm = std::min<size_t>(3, mtop.size());
      std::partial_sort(mtop.begin(), mtop.begin() + nm, mtop.end(), [](const auto& a, const auto& b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
      });
      std::string mtops;
      for (size_t t = 0; t < nm; ++t) {
        const i_t r = mtop[t].first;
        i_t ni = 0, nc = 0, mincr = std::numeric_limits<i_t>::max();
        for (i_t p = fp_host.row_offsets[r]; p < fp_host.row_offsets[r + 1]; ++p) {
          const i_t j = fp_host.row_cols[p];
          if (is_int[j]) {
            ni++;
          } else {
            nc++;
            mincr = std::min(mincr, col_nrows[j]);
          }
        }
        mtops += " row" + std::to_string(fp_host.row_orig[r]) + "(int=" + std::to_string(ni) +
                 " cont=" + std::to_string(nc) + " min_cont_rows=" + std::to_string(mincr) +
                 " lb=" + std::to_string((double)fp_host.row_lb[r]) +
                 " ub=" + std::to_string((double)fp_host.row_ub[r]) +
                 " act=" + std::to_string(mv_act[r]) + ")x" + std::to_string(mtop[t].second);
      }
      CUOPT_LOG_INFO(
        "Slope scaling safe step refusals: open mixed rows with_single_row_col=%d without=%d "
        "distinct_rows=%zu top:%s",
        n_mix_single,
        n_blk[1][1] - n_mix_single,
        mix_rows.size(),
        mtops.c_str());
    }
    if (n_kept == 0 && n_close == 0) { return 5; }
    safe_warm     = true;
    const int res = eval_pattern(f_t(0), 2);
    safe_warm     = false;
    return res;
  };
  // one evaluation of the slope-scaling point at thr: retried after a flip or certificate repair
  // (four times at most)
  auto run_eval = [&](f_t thr) -> int {
    last_eval_obj = std::numeric_limits<f_t>::infinity();
    clear_moves();
    int r = eval_pattern(thr, 0);
    for (int retry = 0; r == 4 && retry < 4; ++retry) {
      r = eval_pattern(thr, 0);
    }
    clear_moves();  // for the retries of this point only
    return r == 4 ? 0 : r;
  };
  // safe steps on the anchored point while each gains at least min_safe_gain (relative; later
  // steps gave ~0.1% per 40 s, a round pattern that hits -0.3..-3.4%); the flips of a step that
  // gains less become tabu, and the chain ends so that the slope-scaling rounds get their turn
  // (until_time: go on while slope scaling has time). Returns 3 when no time is left, 5 when the
  // step would not move the point.
  constexpr f_t min_safe_gain = f_t(3e-3);
  int n_safe_steps            = 0;
  double safe_time = 0, round_time = 0;  // time in safe steps / in rounds and their evaluations
  auto safe_chain  = [&](bool until_time) -> int {
    const double t_start = timer.elapsed_time();
    int rs               = 0;
    while (true) {
      const f_t before = anchor_obj;
      last_eval_obj    = std::numeric_limits<f_t>::infinity();
      rs               = safe_step();
      for (int retry = 0; rs == 4 && retry < 3; ++retry) {
        rs = eval_pattern(f_t(0), 2);  // certificate retries of the safe step's pattern
      }
      clear_moves();
      if (rs == 3 || rs == 5) { break; }
      n_safe_steps++;
      if (anchor_obj < before - min_safe_gain * std::abs(before)) { continue; }
      for (auto v : last_safe) {
        n_tabu += !safe_tabu[v];
        safe_tabu[v] = 1;
      }
      tabu_obj = anchor_obj;
      if (!until_time || last_safe.empty()) { break; }
    }
    safe_time += timer.elapsed_time() - t_start;
    return rs;
  };

  // ---- slope-scaling rounds; a pattern is evaluated after every round from round 3 on, and
  // whenever the open set is stable; a stable duplicate pattern (at every threshold) ends it
  constexpr int max_rounds       = 60;
  constexpr int first_eval_round = 3;
  constexpr int eval_every       = 1;
  const f_t eval_thresholds[]    = {f_t(0.7), f_t(0.5), f_t(0.3)};
  // the round LPs hold the GPU alone (the cut loop yields), so half the former cap gives them as
  // many iterations as before and twice the rounds and pattern attempts
  const double round_cap         = std::min<double>(30., 0.075 * budget);
  // a round starts while it, the evaluation of its pattern and the publication of the result fit
  // in the time left: the last round's time, the longest of the last three evaluations (0.3 budget
  // before the first one, and at most that) and a publish margin
  constexpr f_t publish_margin = 10.;
  double last_round_time       = round_cap;
  auto eval_reserve            = [&]() -> f_t {
    if (eval_times.empty()) { return 0.3 * budget; }
    double longest = 0;
    for (size_t i = eval_times.size() > 3 ? eval_times.size() - 3 : 0; i < eval_times.size(); ++i) {
      longest = std::max(longest, eval_times[i]);
    }
    return std::min<f_t>(0.3 * budget, (f_t)longest + publish_margin);
  };
  std::vector<var_t> h_types(n_vars, var_t::CONTINUOUS);
  int rounds_done = 0;
  for (int round = 1; round <= max_rounds; ++round) {
    if (stop()) { break; }
    if (remaining() < last_round_time + eval_reserve()) {
      CUOPT_LOG_INFO(
        "Slope scaling rounds end: remaining=%.1fs round=%.1fs eval_reserve=%.1fs elapsed=%.2f",
        (double)remaining(),
        last_round_time,
        (double)eval_reserve(),
        timer.elapsed_time());
      break;
    }
    const double t_round       = timer.elapsed_time();
    const double t_round_start = t_round;
    for (size_t k = 0; k < L; ++k) {
      if (u[k] > open_tol(k)) {
        scale[k] = std::clamp<f_t>(u[k], floor_rel * vub[k].cap, vub[k].cap);
      }
    }
    std::vector<f_t> mod_vals = h_vals;
    std::vector<f_t> mod_vub  = h_vub;
    std::vector<f_t> init_x   = cur_x;
    for (size_t k = 0; k < L; ++k) {
      const auto& l   = vub[k];
      const f_t ratio = scale[k] / l.cap;
      mod_vals[l.pos] = h_vals[l.pos] * ratio;
      mod_vub[l.var]  = l.cap / scale[k];
      init_x[l.var]   = std::clamp<f_t>(std::max<f_t>(u[k], 0) / scale[k], 0, mod_vub[l.var]);
    }
    for (i_t j = 0; j < n_vars; ++j) {
      init_x[j] = std::clamp(init_x[j], h_vlb[j], mod_vub[j]);
    }
    auto snapshot = make_root_lp_snapshot(*problem_ptr);
    snapshot.set_csr_constraint_matrix(mod_vals.data(),
                                       (i_t)mod_vals.size(),
                                       h_cols.data(),
                                       (i_t)h_cols.size(),
                                       h_offsets.data(),
                                       (i_t)h_offsets.size());
    snapshot.set_variable_upper_bounds(mod_vub.data(), n_vars);
    snapshot.set_variable_types(h_types.data(), n_vars);
    pdlp_solver_settings_t<i_t, f_t> lp_settings{};
    lp_settings.set_optimality_tolerance(1e-4);
    lp_settings.time_limit =
      std::max<double>(1., std::min<double>(round_cap, remaining() - eval_reserve()));
    lp_settings.method = method_t::PDLP;
    std::atomic<int> local_halt{0};
    lp_settings.concurrent_halt = &local_halt;
    lp_settings.inside_mip      = false;
    lp_settings.num_gpus        = context.settings.num_gpus;
    lp_settings.set_initial_primal_solution(init_x.data(), (i_t)init_x.size(), stream);
    lp_settings.set_initial_dual_solution(cur_y.data(), (i_t)cur_y.size(), stream);
    handle_ptr->sync_stream();
    const double t_lp   = timer.elapsed_time();
    auto lp_result      = solve_lp(snapshot, lp_settings);
    const auto& lp_info = lp_result.get_additional_termination_information();
    const bool usable   = lp_result.get_primal_solution().size() == (size_t)n_vars &&
                        lp_result.get_dual_solution().size() == (size_t)n_cstrs;
    if (!usable) {
      CUOPT_LOG_INFO("Slope scaling round %d: LP status=%s returned no usable point",
                     round,
                     lp_result.get_termination_status_string().c_str());
      break;
    }
    cur_x = cuopt::host_copy(lp_result.get_primal_solution(), stream);
    cur_y = cuopt::host_copy(lp_result.get_dual_solution(), stream);
    rounds_done++;
    last_round_time = timer.elapsed_time() - t_round_start;
    i_t n_open = 0, n_opened = 0, n_closed = 0, n_full = 0;
    double ysum = 0;
    for (size_t k = 0; k < L; ++k) {
      u[k]       = usage(k, cur_x);
      is_open[k] = u[k] > open_tol(k);
      n_open += is_open[k];
      n_opened += is_open[k] && !prev_open[k];
      n_closed += !is_open[k] && prev_open[k];
      if (is_open[k]) {
        const f_t yk = u[k] / scale[k];
        ysum += yk;
        n_full += yk >= 0.5;
      }
    }
    CUOPT_LOG_INFO(
      "Slope scaling round %d: status=%s iters=%d lp_obj=%g l2_primal_res=%g time=%.2fs "
      "open=%d opened=%d closed=%d open_y_mean=%.3f open_y_ge_half=%d elapsed=%.2f",
      round,
      lp_result.get_termination_status_string().c_str(),
      lp_info.number_of_steps_taken,
      (double)lp_result.get_objective_value(),
      lp_info.l2_primal_residual,
      timer.elapsed_time() - t_lp,
      n_open,
      n_opened,
      n_closed,
      n_open > 0 ? ysum / n_open : 0.,
      n_full,
      timer.elapsed_time());
    const bool stable = (n_opened + n_closed) <= std::max<i_t>(1, n_open / 200);
    prev_open         = is_open;
    const bool due    = round >= first_eval_round && (round - first_eval_round) % eval_every == 0;
    if (due || (stable && round >= 2)) {
      // the highest threshold first; the lower ones (more links opened, which the binary rows
      // turn into fix-and-propagate conflicts far more often) only when it repeats a pattern
      int r                 = 1;
      const f_t before_eval = anchor_obj;
      near_obj              = std::numeric_limits<f_t>::infinity();
      for (int t = 0; t < 3 && r == 1; ++t) {
        r = run_eval(eval_thresholds[t]);
      }
      round_time += timer.elapsed_time() - t_round;
      if (r == 3) { break; }
      if (stable && r == 1) { break; }
      // a published round pattern is followed by safe steps while they gain, then the next round.
      // Without a publish the rounds go on undisturbed: a safe step's publish restarts them from
      // its point, and the round patterns of the leader runs hit after a few undisturbed rounds
      if (anchor_obj < before_eval) {
        if (safe_chain(false) == 3) { break; }
      } else if (std::isfinite(near_obj) && std::isfinite(anchor_obj)) {
        // a near miss: safe steps from its point; the incumbent's anchor comes back unless one
        // of them published
        const f_t inc_obj = anchor_obj;
        anchor_x.swap(near_x);
        anchor_y.swap(near_y);
        std::swap(anchor_obj, near_obj);
        const int rs = safe_chain(false);
        if (!(anchor_obj < inc_obj)) {
          anchor_x.swap(near_x);
          anchor_y.swap(near_y);
          anchor_obj = inc_obj;
        }
        CUOPT_LOG_INFO("Slope scaling near-miss chain: %s obj=%g elapsed=%.2f",
                       anchor_obj < inc_obj ? "improved" : "no gain",
                       anchor_obj,
                       timer.elapsed_time());
        near_obj = std::numeric_limits<f_t>::infinity();
        if (rs == 3) { break; }
      }
    } else {
      round_time += timer.elapsed_time() - t_round;
    }
  }
  if (n_evals == 0 && rounds_done > 0) { run_eval(eval_thresholds[0]); }
  // the rounds are over (stable, or the time left is below a round's reserve): safe steps on the
  // anchored point while slope scaling has time
  if (std::isfinite(anchor_obj)) { safe_chain(true); }
  CUOPT_LOG_INFO(
    "Slope scaling end: rounds=%d evaluated=%d published=%d safe_steps=%d time=%.2fs "
    "(rounds %.1fs, safe steps %.1fs) elapsed=%.2f",
    rounds_done,
    n_evals,
    (int)published,
    n_safe_steps,
    timer.elapsed_time() - t0,
    round_time,
    safe_time,
    timer.elapsed_time());
  return published;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_early_fix_propagate()
{
  if (early_fpc_done || problem_ptr->n_integer_vars == 0 ||
      problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return;
  }
  const char* disable_heuristics_env = std::getenv("CUOPT_DISABLE_GPU_HEURISTICS");
  if (disable_heuristics_env != nullptr && std::string(disable_heuristics_env) == "1") { return; }
  if (context.settings.heuristic_params.lp_free_fix_propagate == 0) { return; }
  early_fpc_done       = true;
  population.timer     = timer;
  fpc_external_publish = true;
  CUOPT_LOG_INFO("Early fix-and-propagate start: elapsed=%.2f", timer.elapsed_time());
  run_fix_propagate_complete(false, 25.);
  fpc_external_publish = false;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_lp_guided_fix_propagate()
{
  if (problem_ptr->n_integer_vars == 0 || problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return;
  }
  const auto& hp   = context.settings.heuristic_params;
  const f_t budget = std::min<f_t>(hp.fix_propagate_max_time,
                                   hp.fix_propagate_time_ratio * context.settings.time_limit);
  if (budget <= 1.) { return; }
  CUOPT_LOG_INFO(
    "LP-guided fix-and-propagate start: budget=%.1fs elapsed=%.2f", budget, timer.elapsed_time());
  run_fix_propagate_complete(true, budget);
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_fix_propagate_complete(bool use_lp, f_t max_budget)
{
  raft::common::nvtx::range fun_scope("run_fix_propagate_complete");
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
  // reference point: the root LP optimum, or (before the root LP) zero clamped to the bounds
  std::vector<f_t> h_lp;
  if (use_lp) {
    h_lp = cuopt::host_copy(lp_optimal_solution, stream);
  } else {
    h_lp.resize(n_vars);
    for (i_t j = 0; j < n_vars; ++j) {
      h_lp[j] = std::min(std::max(f_t(0), h_vlb[j]), h_vub[j]);
    }
  }
  host_fix_propagate_t<i_t, f_t> fp_host;
  fp_host.abs_tol = problem_ptr->tolerances.absolute_tolerance;
  fp_host.continuous_min_improvement =
    context.settings.heuristic_params.fix_propagate_continuous_min_improvement;
  fp_host.build(n_vars, n_cstrs, h_offsets, h_cols, h_vals, h_clb, h_cub, h_vlb, h_vub, h_int);
  fp_host.int_tol = problem_ptr->tolerances.integrality_tolerance;
  {
    simplex::user_problem_t<i_t, f_t> user_problem(problem_ptr->handle_ptr);
    problem_ptr->get_host_user_problem(user_problem);
    fp_host.build_domain(user_problem, problem_ptr->clique_table.get());
  }
  i_t n_frac = 0, n_at_one = 0;
  for (auto v : h_int) {
    f_t x = h_lp[v];
    if ((x - std::floor(x) > int_tol) && (std::ceil(x) - x > int_tol)) n_frac++;
    if (x >= 1. - int_tol) n_at_one++;
  }
  CUOPT_LOG_INFO(
    "FPC start: use_lp=%d budget=%.1fs n_int=%zu lp_fractional=%d lp_ge_one=%d int_rows=%d "
    "(int_only=%d mixed=%d) int_row_nnz=%zu build=%.2fs elapsed=%.2f",
    (int)use_lp,
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

  // LP-free order: fractionality (with the zero reference point every integer ties, so this is the
  // index order)
  std::vector<i_t> order(h_int.begin(), h_int.end());
  {
    std::vector<f_t> key(n_vars, 0);
    for (auto v : h_int) {
      key[v] = fp_order_key<i_t, f_t>(
        fp_variable_strategy_t::FRACTIONALITY, h_lp[v], f_t(0), i_t(0), f_t(0));
    }
    std::stable_sort(order.begin(), order.end(), [&](i_t a, i_t b) { return key[a] > key[b]; });
  }

  if (use_lp) {
    // LP-guided fix-and-propagate: one pass with the configured variable / value strategies
    // (see fix_propagate_host.hpp), the pattern evaluated by a short cost-aware fixed LP (its
    // objective settles within a few thousand PDLP iterations even when the residual stalls),
    // then pushed, repaired and published if it improves. All steps are scaled to the budget.
    auto h_obj  = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
    auto h_dual = cuopt::host_copy(lp_dual_optimal_solution, stream);
    std::vector<f_t> red(n_vars, 0);
    std::vector<i_t> uplocks(n_vars, 0);
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
      }
      red[v] = h_obj[v] - aty;
    }
    struct cand_t {
      std::string name;  // log name
      fp_variable_strategy_t var_strategy;
      fp_value_strategy_t value_strategy;
      f_t order_param;  // noise amplitude of LP_VALUE_DESC_NOISE
      f_t value_param;  // reduced-cost or fraction threshold of the value strategy
      unsigned seed;
    };
    std::vector<std::vector<f_t>> seen;
    f_t best_lp = std::numeric_limits<f_t>::infinity();
    int n_eval  = 0;
    std::string best_name;
    std::vector<f_t> best_repaired;
    constexpr f_t rank_tol  = 1e-4;
    const f_t rank_tlimit   = std::min<f_t>(8., 0.25 * budget);
    const f_t push_tlimit   = std::min<f_t>(11., 0.3 * budget);
    const f_t repair_tlimit = std::min<f_t>(20., 0.4 * budget);
    auto run_candidate      = [&](const cand_t& c, const std::vector<f_t>& x) {
      if (fpc_timer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption() ||
          timer.remaining_time() < 2. * budget) {
        return;
      }
      std::vector<i_t> ord(h_int.begin(), h_int.end());
      std::vector<f_t> key(n_vars, 0);
      std::mt19937 g(c.seed);
      std::uniform_real_distribution<double> U(0., 1.);
      const bool noisy = c.var_strategy == fp_variable_strategy_t::LP_VALUE_DESC_NOISE;
      for (auto v : h_int) {
        const f_t noise = noisy ? c.order_param * U(g) : f_t(0);
        key[v]          = fp_order_key<i_t, f_t>(c.var_strategy, x[v], red[v], uplocks[v], noise);
      }
      std::stable_sort(ord.begin(), ord.end(), [&](i_t a, i_t b) { return key[a] > key[b]; });
      std::vector<f_t> pref(n_vars, 0.);
      for (auto v : h_int) {
        pref[v] = fp_preferred_value<f_t>(
          c.value_strategy, x[v], h_vlb[v], h_vub[v], red[v], c.value_param, int_tol);
      }
      const double t_fp = timer.elapsed_time();
      i_t n_flipped = 0, n_implied = 0;
      fp_host.work    = 0;
      i_t n_conflicts = fp_host.fix_and_propagate(ord, pref, n_flipped, n_implied);
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
      CUOPT_LOG_INFO(
        "Fix-and-propagate pass %s: time=%.3fs conflicts=%d flipped=%d implied_off_pref=%d ones=%d "
             "int_cost=%g dup=%d cont_tightenings=%ld elapsed=%.2f",
        c.name.c_str(),
        timer.elapsed_time() - t_fp,
        n_conflicts,
        n_flipped,
        n_implied,
        n_ones,
        int_cost,
        (int)dup,
        (long)fp_host.n_continuous_tightenings,
        timer.elapsed_time());
      if (dup || n_conflicts > 0) { return; }
      seen.push_back(pat);
      std::vector<f_t> assign = x;
      for (size_t q = 0; q < h_int.size(); ++q) {
        assign[h_int[q]] = pat[q];
      }
      auto ev = eval_fixed_pattern(assign,
                                   rank_tol,
                                   rank_tlimit,
                                   nullptr,
                                   nullptr,
                                   true,
                                   repair_tlimit,
                                   "fix_propagate",
                                   best_lp,
                                   push_tlimit);
      n_eval++;
      const bool better = ev.usable && !ev.cutoff && ev.lp_obj < best_lp;
      CUOPT_LOG_INFO(
        "Pattern rank %s: ones=%d lp_obj=%g optimal=%d iters=%d l2_primal_res=%g time=%.2fs "
             "better=%d pushed_lp_obj=%g pushed_res=%g push_time=%.2fs repaired=%g published=%d "
             "elapsed=%.2f",
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
        timer.elapsed_time());
      if (better && !ev.repaired_assignment.empty()) {
        best_lp       = ev.lp_obj;
        best_name     = c.name;
        best_repaired = std::move(ev.repaired_assignment);
      }
    };
    const auto& hp = context.settings.heuristic_params;
    const cand_t configured{"lp_guided",
                            static_cast<fp_variable_strategy_t>(hp.fix_propagate_variable_strategy),
                            static_cast<fp_value_strategy_t>(hp.fix_propagate_value_strategy),
                            hp.fix_propagate_order_noise,
                            hp.fix_propagate_value_param,
                            12u};
    run_candidate(configured, h_lp);
    if (best_repaired.empty()) {
      CUOPT_LOG_INFO("LP-guided fix-and-propagate: no repaired pattern (evaluated %d) elapsed=%.2f",
                     n_eval,
                     timer.elapsed_time());
      return false;
    }
    CUOPT_LOG_INFO("LP-guided fix-and-propagate best: %s lp_obj=%g evaluated=%d elapsed=%.2f",
                   best_name.c_str(),
                   best_lp,
                   n_eval,
                   timer.elapsed_time());
    // the pattern was already pushed, repaired and published by its evaluation
    return true;
  }

  // LP-free attempts: every integer at its upper bound, then at its lower bound
  const fp_value_strategy_t lp_free_values[] = {fp_value_strategy_t::UPPER_BOUND,
                                                fp_value_strategy_t::LOWER_BOUND};
  for (int attempt = 0; attempt < 2; ++attempt) {
    if (fpc_timer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) {
      break;
    }
    std::vector<f_t> pref(n_vars, 0.);
    for (auto v : h_int) {
      pref[v] = fp_preferred_value<f_t>(
        lp_free_values[attempt], h_lp[v], h_vlb[v], h_vub[v], f_t(0), f_t(0), int_tol);
    }
    const double t_fp = timer.elapsed_time();
    i_t n_flipped = 0, n_implied = 0;
    fp_host.work    = 0;
    i_t n_conflicts = fp_host.fix_and_propagate(order, pref, n_flipped, n_implied);
    i_t n_dead      = 0;
    for (i_t r = 0; r < fp_host.n_rows(); ++r) {
      n_dead += fp_host.row_dead[r];
    }
    i_t n_ones = 0, n_diff = 0;
    for (auto v : h_int) {
      n_ones += fp_host.col_lb[v] > 0.5;
      n_diff += std::abs(fp_host.col_lb[v] - std::round(h_lp[v])) > 0.5;
    }
    solution_t<i_t, f_t> sol(*problem_ptr);
    {
      std::vector<f_t> h_assign = h_lp;
      for (auto v : h_int) {
        h_assign[v] = fp_host.col_lb[v];
      }
      sol.copy_new_assignment(h_assign);
    }
    bool feas = sol.compute_feasibility();
    CUOPT_LOG_INFO(
      "FPC attempt %d fix-and-propagate pass: time=%.3fs conflicts=%d dead_rows=%d flipped=%d "
      "implied_off_pref=%d "
      "ones=%d differs_from_lp_nearest=%d work=%ld feasible=%d max_cstr_viol=%g elapsed=%.2f",
      attempt,
      timer.elapsed_time() - t_fp,
      n_conflicts,
      n_dead,
      n_flipped,
      n_implied,
      n_ones,
      n_diff,
      (long)fp_host.work,
      (int)feas,
      sol.compute_max_constraint_violation(),
      timer.elapsed_time());
    if (!feas) { feas = fpc_complete_continuous(sol, fpc_timer, attempt, true); }
    if (feas) {
      CUOPT_LOG_INFO("FPC attempt %d found feasible solution obj=%g elapsed=%.2f",
                     attempt,
                     sol.get_user_objective(),
                     timer.elapsed_time());
      if (fpc_external_publish) {
        population.add_external_solution(
          sol.get_host_assignment(), sol.get_objective(), solution_origin_t::FIX_PROPAGATE);
      } else {
        population.add_solution(std::move(sol), "fix_propagate_complete_nolp");
      }
      return true;
    }
  }
  CUOPT_LOG_INFO("FPC end without feasible solution elapsed=%.2f", timer.elapsed_time());
  return false;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::add_user_given_solutions(
  std::vector<solution_t<i_t, f_t>>& initial_sol_vector)
{
  raft::common::nvtx::range fun_scope("add_user_given_solutions");
  const bool has_papilo            = problem_ptr->has_papilo_presolve_data();
  const i_t papilo_orig_n          = problem_ptr->get_papilo_original_num_variables();
  auto add_original_space_solution = [&](rmm::device_uvector<f_t> init_sol_assignment,
                                         size_t sol_idx) {
    solution_t<i_t, f_t> sol(*problem_ptr);

    if (has_papilo) {
      if ((i_t)init_sol_assignment.size() != papilo_orig_n) {
        CUOPT_LOG_ERROR(
          "add the provided initial solution! Initial solution %zu has %zu vars, "
          "expected %d; skipping",
          sol_idx,
          init_sol_assignment.size(),
          papilo_orig_n);
        return;
      }
      std::vector<f_t> h_original = host_copy(init_sol_assignment, sol.handle_ptr->get_stream());
      std::vector<f_t> h_crushed;
      const auto* presolver_ptr = problem_ptr->presolve_data.papilo_presolve_ptr;
      presolver_ptr->crush_primal_solution(
        *problem_ptr->original_problem_ptr, h_original, h_crushed);
      init_sol_assignment = cuopt::device_copy(h_crushed, sol.handle_ptr->get_stream());

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG
      const auto& reduced_problem       = *problem_ptr->original_problem_ptr;
      const std::vector<f_t> h_red_obj  = reduced_problem.get_objective_coefficients_host();
      const std::vector<f_t>& h_ori_obj = presolver_ptr->get_original_objective_coefficients();
      cuopt_assert(h_ori_obj.size() == h_original.size(),
                   "original objective size must match input solution dimension");
      cuopt_assert(h_red_obj.size() == h_crushed.size(),
                   "reduced objective size must match crushed solution dimension");
      // Map each solution to user space with its own problem's scale, so the comparison holds even
      // if the original and reduced objective scales ever diverge.
      [[maybe_unused]] const double input_obj =
        (double)presolver_ptr->get_original_objective_scaling_factor() *
        std::inner_product(h_ori_obj.begin(),
                           h_ori_obj.end(),
                           h_original.begin(),
                           (double)presolver_ptr->get_original_objective_offset());
      [[maybe_unused]] const double crushed_obj =
        (double)reduced_problem.get_objective_scaling_factor() *
        std::inner_product(h_red_obj.begin(),
                           h_red_obj.end(),
                           h_crushed.begin(),
                           (double)reduced_problem.get_objective_offset());
      CUOPT_LOG_DEBUG(
        "Crushed initial solution %d through Papilo (%d -> %d vars), objective %g -> %g",
        sol_idx,
        papilo_orig_n,
        h_crushed.size(),
        input_obj,
        crushed_obj);
#endif
    }

    if (problem_ptr->pre_process_assignment(init_sol_assignment)) {
      raft::copy(sol.assignment.data(),
                 init_sol_assignment.data(),
                 init_sol_assignment.size(),
                 sol.handle_ptr->get_stream());
      relaxed_lp_settings_t lp_settings;
      lp_settings.time_limit            = std::min(60., timer.remaining_time() / 2);
      lp_settings.tolerance             = problem_ptr->tolerances.absolute_tolerance;
      lp_settings.save_state            = false;
      lp_settings.return_first_feasible = true;
      run_lp_with_vars_fixed(*problem_ptr,
                             sol,
                             problem_ptr->integer_indices,
                             lp_settings,
                             static_cast<bound_presolve_t<i_t, f_t>*>(nullptr));
      bool is_feasible = sol.compute_feasibility();
      if (!is_feasible) {
        raft::copy(sol.assignment.data(),
                   init_sol_assignment.data(),
                   init_sol_assignment.size(),
                   sol.handle_ptr->get_stream());
        is_feasible = sol.compute_feasibility();
      }

      cuopt_func_call(sol.test_variable_bounds(true));
      CUOPT_LOG_DEBUG("Adding initial solution success! feas %d objective %f excess %f",
                      is_feasible,
                      sol.get_user_objective(),
                      sol.get_total_excess());
      population.run_solution_callbacks(sol, "user_initial_solution");
      initial_sol_vector.emplace_back(std::move(sol));
    } else {
      CUOPT_LOG_ERROR(
        "Error cannot add the provided initial solution! Assignment size %lu initial solution size "
        "%lu",
        sol.assignment.size(),
        init_sol_assignment.size());
    }
  };
  const auto stream = problem_ptr->handle_ptr->get_stream();
  for (size_t sol_idx = 0; sol_idx < context.settings.initial_solutions.size(); ++sol_idx) {
    if (timer.check_time_limit()) { return; }
    add_original_space_solution(
      rmm::device_uvector<f_t>(*context.settings.initial_solutions[sol_idx], stream), sol_idx);
  }
  if (context.initial_incumbent_from_papilo_model && !timer.check_time_limit()) {
    add_original_space_solution(cuopt::device_copy(context.initial_incumbent_assignment, stream),
                                context.settings.initial_solutions.size());
  }
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_presolve(f_t time_limit, timer_t global_timer)
{
  raft::common::nvtx::range fun_scope("run_presolve");
  CUOPT_LOG_INFO("\nRunning cuOpt presolve");
  timer_t presolve_timer(time_limit);

  const f_t bounds_time_limit = std::min({ls.constraint_prop.bounds_update.settings.time_limit,
                                          (f_t)presolve_timer.remaining_time(),
                                          (f_t)global_timer.remaining_time()});
  if (bounds_time_limit <= 0.0) {
    stats.presolve_time = presolve_timer.elapsed_time();
    return true;
  }
  ls.constraint_prop.bounds_update.settings.time_limit = bounds_time_limit;
  auto term_crit = ls.constraint_prop.bounds_update.solve(*problem_ptr);
  if (ls.constraint_prop.bounds_update.infeas_constraints_count > 0) {
    stats.presolve_time = timer.elapsed_time();
    return false;
  }
  if (termination_criterion_t::NO_UPDATE != term_crit) {
    ls.constraint_prop.bounds_update.set_updated_bounds(*problem_ptr);
  }
  const auto& hp              = context.settings.heuristic_params;
  const auto probing_features = probing_presolve_features(*problem_ptr);
  const auto probing_budget   = evaluate_presolve_budget(hp, probing_features);
  bool run_probing_cache      = !fj_only_run;
  // Allow the user to disable the probing-cache step of cuOpt's internal presolve
  // independently of the higher-level presolver setting.
  if (!context.settings.probing) {
    CUOPT_LOG_INFO("Probing-cache step disabled via %s=false", CUOPT_MIP_PROBING);
    run_probing_cache = false;
  }
  const bool remap_cache_ids           = true;
  problem_ptr->related_vars_time_limit = context.settings.heuristic_params.related_vars_time_limit;

  if (run_probing_cache && !global_timer.check_time_limit() && !presolve_timer.check_time_limit()) {
    log_presolve_budget("PROBING", probing_features, probing_budget);
    // The early CPUFJ lanes hold their threads for the whole of presolve, and probing's default
    // task count assumes the whole team. Its pools are sized per task, so this bounds host memory
    // as well as concurrency.
    const i_t held_by_cpufj =
      context.early_cpufj_ptr != nullptr ? (i_t)context.early_cpufj_ptr->lane_count() : 0;
    ls.constraint_prop.bounds_update.settings.num_tasks = probing_thread_budget(
      omp_get_num_threads(), held_by_cpufj, context.early_structural_ptr != nullptr ? 1 : 0);
    f_t time_for_probing_cache = std::min(time_limit, (f_t)global_timer.remaining_time());
    timer_t probing_timer{time_for_probing_cache};
    [[maybe_unused]] const auto probing_t0 = std::chrono::steady_clock::now();
    // this function computes probing cache, finds singletons, substitutions and changes the problem
    bool problem_is_infeasible = compute_probing_cache(ls.constraint_prop.bounds_update,
                                                       *problem_ptr,
                                                       probing_timer,
                                                       probing_budget.probing_work_limit,
                                                       (size_t)probing_budget.probing_step_size);
    problem_ptr->handle_ptr->sync_stream();
    CUOPT_LOG_INFO(
      "PRESOLVE_PROBING_WALL wall=%.3f presolve_elapsed=%.3f",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - probing_t0).count(),
      presolve_timer.elapsed_time());
    if (problem_is_infeasible) { return false; }
  }

  if (!global_timer.check_time_limit()) { trivial_presolve(*problem_ptr, remap_cache_ids); }

  if (context.settings.block_bve && run_probing_cache) {
    timer_t bve_deadline(std::min(global_timer.remaining_time(), presolve_timer.remaining_time()));
    if (!block_bve_phase(ls.constraint_prop.bounds_update, *problem_ptr, bve_deadline)) {
      stats.presolve_time = timer.elapsed_time();
      return false;
    }
  }

  if (!problem_ptr->empty && !check_bounds_sanity(*problem_ptr)) { return false; }
  // if (!presolve_timer.check_time_limit() && !context.settings.heuristics_only &&
  //     !problem_ptr->empty) {
  //   f_t time_limit_for_clique_table = std::min(3., presolve_timer.remaining_time() / 5);
  //   timer_t clique_timer(time_limit_for_clique_table);
  //   simplex::user_problem_t<i_t, f_t> host_problem(problem_ptr->handle_ptr);
  //   problem_ptr->get_host_user_problem(host_problem);
  //   std::shared_ptr<clique_table_t<i_t, f_t>> clique_table;
  //   constexpr bool modify_problem_with_cliques = false;
  //   find_initial_cliques(host_problem,
  //                        context.settings.tolerances,
  //                        &clique_table,
  //                        clique_timer,
  //                        modify_problem_with_cliques,
  //                        nullptr);
  //   if (modify_problem_with_cliques) {
  //     problem_ptr->set_constraints_from_host_user_problem(host_problem);
  //     cuopt_assert(host_problem.lower.size() == static_cast<size_t>(problem_ptr->n_variables),
  //                  "host lower bound size mismatch");
  //     cuopt_assert(host_problem.upper.size() == static_cast<size_t>(problem_ptr->n_variables),
  //                  "host upper bound size mismatch");
  //     std::vector<i_t> all_var_indices(problem_ptr->n_variables);
  //     std::iota(all_var_indices.begin(), all_var_indices.end(), 0);
  //     problem_ptr->update_variable_bounds(all_var_indices, host_problem.lower,
  //     host_problem.upper); trivial_presolve(*problem_ptr, remap_cache_ids);
  //   }
  // }
  // May overconstrain if Papilo presolve has been run before
  if (context.settings.presolver == presolver_t::None) {
    if (!problem_ptr->empty) {
      // do the resizing no-matter what, bounds presolve might not change the bounds but initial
      // trivial presolve might have
      ls.constraint_prop.bounds_update.resize(*problem_ptr);
      ls.constraint_prop.bounds_update.upd.init_changed_constraints(problem_ptr->handle_ptr);
      ls.constraint_prop.conditional_bounds_update.update_constraint_bounds(
        *problem_ptr, ls.constraint_prop.bounds_update);
    }
    if (!check_bounds_sanity(*problem_ptr)) { return false; }
  }
  stats.presolve_time = presolve_timer.elapsed_time();
  lp_optimal_solution.resize(problem_ptr->n_variables, problem_ptr->handle_ptr->get_stream());
  lp_dual_optimal_solution.resize(problem_ptr->n_constraints,
                                  problem_ptr->handle_ptr->get_stream());
  problem_ptr->handle_ptr->sync_stream();
  CUOPT_LOG_INFO("After cuOpt presolve: %d constraints, %d variables, objective offset %f.",
                 problem_ptr->n_constraints,
                 problem_ptr->n_variables,
                 problem_ptr->presolve_data.objective_offset);
  CUOPT_LOG_INFO("cuOpt presolve time: %.2f", stats.presolve_time);
  return true;
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::generate_quick_feasible_solution()
{
  raft::common::nvtx::range fun_scope("generate_quick_feasible_solution");
  solution_t<i_t, f_t> solution(*problem_ptr);
  // min 1 second, max 10 seconds
  const f_t generate_fast_solution_time =
    std::min(diversity_config.max_fast_sol_time, std::max(1., timer.remaining_time() / 20.));
  timer_t sol_timer(generate_fast_solution_time);
  // do very short LP run to get somewhere close to the optimal point
  ls.generate_fast_solution(solution, sol_timer);
  if (solution.get_feasible()) {
    population.run_solution_callbacks(solution, "quick_feasible");
    initial_sol_vector.emplace_back(std::move(solution));
    problem_ptr->handle_ptr->sync_stream();
    solution_t<i_t, f_t> searched_sol(initial_sol_vector.back());
    ls_config_t<i_t, f_t> ls_config;
    run_local_search(searched_sol, population.weights, sol_timer, ls_config);
    population.run_solution_callbacks(searched_sol, "quick_feasible_local_search");
    initial_sol_vector.emplace_back(std::move(searched_sol));
    auto& feas_sol = initial_sol_vector.back().get_feasible()
                       ? initial_sol_vector.back()
                       : initial_sol_vector[initial_sol_vector.size() - 2];
    CUOPT_LOG_INFO("Generated fast solution in %f seconds with objective %f",
                   timer.elapsed_time(),
                   feas_sol.get_user_objective());
  }
  problem_ptr->handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::check_b_b_preemption()
{
  if (context.preempt_heuristic_solver_.load()) {
    if (population.current_size() == 0) { population.allocate_solutions(); }
    population.add_external_solutions_to_population();
    return true;
  }
  population.add_external_solutions_to_population();
  return false;
}

// returns the best feasible solution
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_fj_alone(solution_t<i_t, f_t>& solution)
{
  CUOPT_LOG_INFO("Running FJ alone!");
  solution.round_nearest(rng());
  ls.fj.settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  ls.fj.settings.n_of_minimums_for_exit = 20000 * 1000;
  ls.fj.settings.update_weights         = true;
  ls.fj.settings.feasibility_run        = false;
  ls.fj.settings.time_limit             = timer.remaining_time();
  ls.fj.solve(solution, ls.make_fj_improvement_callback(solution.problem_ptr));
  CUOPT_LOG_INFO("FJ alone finished!");
}

// returns the best feasible solution
template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::run_fp_alone()
{
  CUOPT_LOG_DEBUG("Running FP alone!");
  solution_t<i_t, f_t> sol(population.best_feasible());
  ls.run_fp(sol, timer, &population);
  CUOPT_LOG_DEBUG("FP alone finished!");
}

template <typename i_t, typename f_t>
struct ls_cpufj_raii_guard_t {
  ls_cpufj_raii_guard_t(local_search_t<i_t, f_t>& ls) : ls(ls) {}
  ~ls_cpufj_raii_guard_t() { ls.stop_cpufj_scratch_threads(); }
  local_search_t<i_t, f_t>& ls;
};

// returns the best feasible solution
template <typename i_t, typename f_t>
solution_t<i_t, f_t> diversity_manager_t<i_t, f_t>::run_solver()
{
  raft::common::nvtx::range fun_scope("run_solver");

  CUOPT_LOG_DEBUG("Determinism mode: %s",
                  context.settings.determinism_mode == CUOPT_MODE_DETERMINISTIC ? "deterministic"
                                                                                : "opportunistic");

  // to automatically compute the solving time on scope exit
  auto timer_raii_guard =
    cuopt::scope_guard([&]() { stats.total_solve_time = timer.elapsed_time(); });

  // Debug: Allow disabling GPU heuristics to test B&B tree determinism in isolation
  const char* disable_heuristics_env = std::getenv("CUOPT_DISABLE_GPU_HEURISTICS");
  if (context.settings.determinism_mode == CUOPT_MODE_DETERMINISTIC) {
    CUOPT_LOG_INFO("Running deterministic mode with CPUFJ heuristic");
    population.initialize_population();
    population.allocate_solutions();

    // Start CPUFJ in deterministic mode with B&B integration
    if (context.branch_and_bound_ptr != nullptr) {
      ls.start_cpufj_deterministic(*context.branch_and_bound_ptr);
    }

    while (!check_b_b_preemption()) {
      if (timer.check_time_limit()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop CPUFJ when B&B is done
    ls.stop_cpufj_deterministic();

    population.add_external_solutions_to_population();
    return population.best_feasible();
  }
  if (disable_heuristics_env != nullptr && std::string(disable_heuristics_env) == "1") {
    CUOPT_LOG_INFO("GPU heuristics disabled via CUOPT_DISABLE_GPU_HEURISTICS=1");
    population.initialize_population();
    population.allocate_solutions();
    add_user_given_solutions(initial_sol_vector);
    population.add_solutions_from_vec(std::move(initial_sol_vector), "initial_solutions");
    if (check_b_b_preemption()) { return population.best_feasible(); }

    while (!check_b_b_preemption()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return population.best_feasible();
  }

  population.timer     = timer;
  const f_t time_limit = timer.remaining_time();
  const auto& hp       = context.settings.heuristic_params;
  const bool slope_scaling_ahead =
    hp.fix_propagate != 0 && !fj_only_run && has_slope_scaling_links();
  // with slope scaling ahead the root LP point is only its starting point (its first round
  // re-solves the LP from it) and PDLP sits on a residual plateau long before 60 s on large models
  constexpr f_t slope_scaling_root_lp_time = 30.;
  const f_t lp_time_limit =
    std::min({hp.root_lp_max_time,
              time_limit * hp.root_lp_time_ratio,
              slope_scaling_ahead ? slope_scaling_root_lp_time : std::numeric_limits<f_t>::infinity()});
  // after every change to the problem, we should resize all the relevant vars
  // we need to encapsulate that to prevent repetitions
  recombine_stats.reset();
  ls.resize_vectors(*problem_ptr, problem_ptr->handle_ptr);
  ls.constraint_prop.bounds_update.resize(*problem_ptr);
  problem_ptr->check_problem_representation(true);
  // have the structure ready for reusing later
  problem_ptr->compute_integer_fixed_problem();
  recombiner_t<i_t, f_t>::init_enabled_recombiners(
    *problem_ptr, context.settings.heuristic_params.enabled_recombiners);
  mab_recombiner.resize_mab_arm_stats(recombiner_t<i_t, f_t>::enabled_recombiners.size());
  // test problem is not ii
  cuopt_func_call(
    ls.constraint_prop.bounds_update.calculate_activity_on_problem_bounds(*problem_ptr));
  cuopt_assert(
    ls.constraint_prop.bounds_update.calculate_infeasible_redundant_constraints(*problem_ptr),
    "The problem must not be ii");
  CUOPT_LOG_INFO("Heuristics setup: structures elapsed=%.2f", timer.elapsed_time());
  population.initialize_population();
  population.allocate_solutions();
  add_user_given_solutions(initial_sol_vector);
  population.add_solutions_from_vec(std::move(initial_sol_vector), "initial_solutions");
  CUOPT_LOG_INFO("Heuristics setup: population elapsed=%.2f", timer.elapsed_time());
  if (check_b_b_preemption()) { return population.best_feasible(); }
  // Run CPUFJ early to find quick initial solutions
  ls_cpufj_raii_guard_t ls_cpufj_raii_guard(ls);  // RAII to stop cpufj threads on solve stop
  ls.start_cpufj_scratch_threads(population);
  CUOPT_LOG_INFO("Heuristics setup: scratch CPUFJ started elapsed=%.2f", timer.elapsed_time());
  // Dedicated ruin-and-repair improvement worker on a spare thread; deepens the population's
  // feasible incumbent without taking any thread away from feasibility discovery above.
  ls.start_cpufj_lns_improvement_thread(population);

  if (check_b_b_preemption()) { return population.best_feasible(); }
  CUOPT_LOG_INFO("Heuristics start (before root LP): elapsed=%.2f", timer.elapsed_time());
  // LP-free fix-and-propagate attempt before the root LP (bounded, objective-free completion)
  const bool lp_free_fp = context.settings.heuristic_params.lp_free_fix_propagate != 0;
  if (lp_free_fp && !fj_only_run && !early_fpc_done && !simplex_solution_exists.load() &&
      !check_b_b_preemption()) {
    run_fix_propagate_complete(false, 25.);
    population.add_external_solutions_to_population();
    if (check_b_b_preemption()) { return population.best_feasible(); }
  }
  // cost-aware polish of the LP-free incumbent before the root LP. Not when slope scaling will
  // run after the root LP: its pattern evaluations bring their own polish, and this one would
  // delay them (tens of seconds on large models).
  if (lp_free_fp && !fj_only_run && !simplex_solution_exists.load() && !check_b_b_preemption()) {
    population.add_external_solutions_to_population();
    if (slope_scaling_ahead) {
      CUOPT_LOG_INFO("Early polish skipped: slope scaling links found, elapsed=%.2f",
                     timer.elapsed_time());
    }
    if (population.is_feasible() && !slope_scaling_ahead) {
      auto early_best = population.best_feasible();
      // stage 0 only, short cap: the root LP (and the LP-guided patterns) start sooner
      polish_continuous(early_best, 40., nullptr, 0, nullptr, nullptr, 0, 10.);
    }
    if (check_b_b_preemption()) { return population.best_feasible(); }
  }
  lp_state_t<i_t, f_t>& lp_state = problem_ptr->lp_state;
  // resize because some constructor might be called before the presolve
  lp_state.resize(*problem_ptr, problem_ptr->handle_ptr->get_stream());
  bool bb_thread_solution_exists = simplex_solution_exists.load();
  if (bb_thread_solution_exists) {
    consume_staged_simplex_solution(lp_state);
    ls.lp_optimal_exists = true;
  } else if (!fj_only_run) {
    convert_greater_to_less(*problem_ptr);

    pdlp_solver_settings_t<i_t, f_t> pdlp_settings{};
    pdlp_settings.time_limit            = lp_time_limit;
    pdlp_settings.first_primal_feasible = false;
    pdlp_settings.concurrent_halt       = &global_concurrent_halt;
    pdlp_settings.method                = context.settings.method;
    pdlp_settings.inside_mip            = false;
    pdlp_settings.num_gpus              = context.settings.num_gpus;
    if (context.settings.heuristic_params.root_pdlp_dual_bound != 0 &&
        problem_ptr->set_root_lower_bound_callback != nullptr) {
      // PDLP may solve a presolved copy of the root LP: it reports in user space, which both share.
      pdlp_settings.dual_bound_frequency =
        context.settings.heuristic_params.root_pdlp_dual_bound_frequency;
      pdlp_settings.dual_bound_callback = [this](f_t user_bound) {
        problem_ptr->set_root_lower_bound_callback(
          problem_ptr->get_solver_obj_from_user_obj(user_bound));
      };
    }
    timer_t lp_timer(lp_time_limit);
    auto root_lp_snapshot    = make_root_lp_snapshot(*problem_ptr);
    pdlp_settings.time_limit = lp_timer.remaining_time();
    CUOPT_LOG_INFO(
      "Root LP call started: method=%s budget=%.3fs mip_elapsed=%.3fs pdlp_mode=%d "
      "inside_mip=%d presolver=Default per_constraint=%d snapshot_seconds=%.3f",
      method_to_string(pdlp_settings.method).c_str(),
      lp_time_limit,
      timer.elapsed_time(),
      static_cast<int>(pdlp_settings.pdlp_solver_mode),
      static_cast<int>(pdlp_settings.inside_mip),
      static_cast<int>(pdlp_settings.per_constraint_residual),
      lp_timer.elapsed_time());
    auto lp_result          = solve_lp(root_lp_snapshot, pdlp_settings);
    const auto root_lp_info = lp_result.get_additional_termination_informations();
    const auto root_lp_method =
      root_lp_info.empty() ? method_t::Unset : root_lp_info.front().solved_by;
    CUOPT_LOG_INFO("Root LP call completed: status=%s method=%s elapsed=%.3fs budget=%.3fs",
                   lp_result.get_termination_status_string().c_str(),
                   method_to_string(root_lp_method).c_str(),
                   lp_timer.elapsed_time(),
                   lp_time_limit);
    CUOPT_LOG_INFO(
      "Root LP postsolve mapping: primal=%zu expected_primal=%zu dual=%zu "
      "expected_dual=%zu",
      lp_result.get_primal_solution().size(),
      lp_optimal_solution.size(),
      lp_result.get_dual_solution().size(),
      lp_dual_optimal_solution.size());
    const auto root_lp_status = lp_result.get_termination_status();

    // The concurrent root LP can fail to produce a usable solution -- e.g. the barrier
    // hits a numerical error on an infeasible problem and PDLP returns NumericalError
    // with empty primal/dual. In that case we must not copy or hand off the empty
    // result (copying n elements from an empty buffer throws), and we must still
    // release B&B's root-relaxation wait so it proceeds with its own dual-simplex root
    // instead of spinning forever.
    const bool usable_root_lp_status =
      root_lp_status == pdlp_termination_status_t::Optimal ||
      root_lp_status == pdlp_termination_status_t::PrimalFeasible ||
      root_lp_status == pdlp_termination_status_t::IterationLimit ||
      root_lp_status == pdlp_termination_status_t::TimeLimit ||
      root_lp_status == pdlp_termination_status_t::ConcurrentLimit;
    const bool root_lp_usable =
      usable_root_lp_status &&
      lp_result.get_primal_solution().size() == lp_optimal_solution.size() &&
      lp_result.get_dual_solution().size() == lp_dual_optimal_solution.size() &&
      lp_result.get_reduced_cost().size() == lp_optimal_solution.size();
    CUOPT_LOG_INFO(
      "GPU root LP status %s, budget %.2f s, elapsed %.2f s, iterations %d, "
      "hit_time_limit %d, usable_point %d",
      lp_result.get_termination_status_string().c_str(),
      lp_time_limit,
      lp_timer.elapsed_time(),
      lp_result.get_additional_termination_information().number_of_steps_taken,
      root_lp_status == pdlp_termination_status_t::TimeLimit ? 1 : 0,
      root_lp_usable ? 1 : 0);

    bool use_staged_simplex_solution = false;
    {
      std::lock_guard<std::mutex> guard(relaxed_solution_mutex);
      use_staged_simplex_solution = simplex_solution_exists.load();
      if (!use_staged_simplex_solution && root_lp_usable) {
        raft::copy(lp_optimal_solution.data(),
                   lp_result.get_primal_solution().data(),
                   lp_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
        raft::copy(lp_dual_optimal_solution.data(),
                   lp_result.get_dual_solution().data(),
                   lp_dual_optimal_solution.size(),
                   problem_ptr->handle_ptr->get_stream());
      }
    }
    if (use_staged_simplex_solution) { consume_staged_simplex_solution(lp_state); }
    if (use_staged_simplex_solution || root_lp_usable) {
      cuopt_assert(thrust::all_of(problem_ptr->handle_ptr->get_thrust_policy(),
                                  lp_optimal_solution.begin(),
                                  lp_optimal_solution.end(),
                                  [] __host__ __device__(f_t val) { return std::isfinite(val); }),
                   "LP optimal solution contains non-finite values");
    }
    ls.lp_optimal_exists = true;
    if (!use_staged_simplex_solution) {
      if (!root_lp_usable) {
        // The concurrent root LP produced no usable solution. Do not hand an empty
        // solution to B&B; instead release its root-relaxation wait loop so it falls
        // back to its own dual-simplex root rather than deadlocking.
        CUOPT_LOG_DEBUG("Root LP produced no usable solution (status %d); releasing B&B root solve",
                        (int)lp_result.get_termination_status());
        ls.lp_optimal_exists = false;
        if (context.branch_and_bound_ptr != nullptr) {
          context.branch_and_bound_ptr->set_root_concurrent_halt(1);
        }
      } else if (lp_result.get_termination_status() == pdlp_termination_status_t::Optimal) {
        solution_t<i_t, f_t> lp_sol(*problem_ptr);
        lp_sol.copy_new_assignment(lp_optimal_solution);
        const bool consider_integrality = false;
        lp_sol.compute_feasibility(consider_integrality);
        if (lp_sol.get_feasible()) { set_new_user_bound(lp_result.get_objective_value()); }
      } else if (lp_result.get_termination_status() ==
                 pdlp_termination_status_t::PrimalInfeasible) {
        CUOPT_LOG_ERROR("Problem is primal infeasible, continuing anyway!");
        ls.lp_optimal_exists = false;
      } else if (lp_result.get_termination_status() == pdlp_termination_status_t::DualInfeasible) {
        CUOPT_LOG_ERROR("PDLP detected dual infeasibility, continuing anyway!");
        ls.lp_optimal_exists = false;
      } else if (lp_result.get_termination_status() == pdlp_termination_status_t::TimeLimit) {
        CUOPT_LOG_DEBUG(
          "Initial LP run exceeded time limit, continuing solver with partial LP result!");
        // note to developer, in debug mode the LP run might be too slow and it might cause PDLP
        // not to bring variables within the bounds
      }
    }

    // Hand the root relaxation off to branch and bound when we have a usable solution
    // (sets root_crossover_solution_set_, releasing B&B's wait). When the root LP failed
    // the wait is instead released above via set_root_concurrent_halt, and a staged
    // dual-simplex solution is owned by B&B already, so neither needs this hand-off.
    if (!use_staged_simplex_solution && root_lp_usable &&
        problem_ptr->set_root_relaxation_solution_callback != nullptr) {
      auto& d_primal_solution = lp_result.get_primal_solution();
      auto& d_dual_solution   = lp_result.get_dual_solution();
      auto& d_reduced_costs   = lp_result.get_reduced_cost();

      std::vector<f_t> host_primal(d_primal_solution.size());
      std::vector<f_t> host_dual(d_dual_solution.size());
      std::vector<f_t> host_reduced_costs(d_reduced_costs.size());
      raft::copy(host_primal.data(),
                 d_primal_solution.data(),
                 d_primal_solution.size(),
                 problem_ptr->handle_ptr->get_stream());
      raft::copy(host_dual.data(),
                 d_dual_solution.data(),
                 d_dual_solution.size(),
                 problem_ptr->handle_ptr->get_stream());
      raft::copy(host_reduced_costs.data(),
                 d_reduced_costs.data(),
                 d_reduced_costs.size(),
                 problem_ptr->handle_ptr->get_stream());
      problem_ptr->handle_ptr->sync_stream();

      // PDLP returns user-space objective (it applies objective_scaling_factor internally)
      auto user_obj   = lp_result.get_objective_value();
      auto solver_obj = problem_ptr->get_solver_obj_from_user_obj(user_obj);
      auto iterations = lp_result.get_additional_termination_information().number_of_steps_taken;
      auto method     = lp_result.get_additional_termination_information().solved_by;
      // Set for the B&B (param4 expects solver space, param5 expects user space)
      problem_ptr->set_root_relaxation_solution_callback(host_primal,
                                                         host_dual,
                                                         host_reduced_costs,
                                                         solver_obj,
                                                         user_obj,
                                                         iterations,
                                                         method,
                                                         root_lp_status);
    }

    if (!use_staged_simplex_solution && root_lp_usable) {
      // in case the pdlp returned var boudns that are out of bounds
      clamp_within_var_bounds(lp_optimal_solution, problem_ptr, problem_ptr->handle_ptr);
    }
  }

  if (ls.lp_optimal_exists) {
    solution_t<i_t, f_t> lp_rounded_sol(*problem_ptr);
    lp_rounded_sol.copy_new_assignment(lp_optimal_solution);
    lp_rounded_sol.round_nearest(rng());
    lp_rounded_sol.compute_feasibility();
    population.add_solution(std::move(lp_rounded_sol), "lp_round_nearest");
    CUOPT_LOG_INFO("Heuristics setup: LP rounding added elapsed=%.2f", timer.elapsed_time());
    ls.start_cpufj_lptopt_scratch_threads(population);
    CUOPT_LOG_INFO("Heuristics setup: LP-start CPUFJ started elapsed=%.2f", timer.elapsed_time());
  }

  if (check_b_b_preemption()) { return population.best_feasible(); }

  if (context.settings.benchmark_info_ptr != nullptr) {
    context.settings.benchmark_info_ptr->objective_of_initial_population =
      population.best_feasible().get_user_objective();
  }

  if (fj_only_run) {
    solution_t<i_t, f_t> sol(*problem_ptr);
    run_fj_alone(sol);
    return sol;
  }

  if (context.settings.heuristic_params.fix_propagate != 0) {
    ls.lp_point_fix_propagate = [this]() {
      CUOPT_LOG_INFO("Heuristics setup: LP point reached elapsed=%.2f", timer.elapsed_time());
      run_lp_guided_fix_propagate();
      CUOPT_LOG_INFO("Heuristics setup: LP-guided FPC done elapsed=%.2f", timer.elapsed_time());
      population.add_external_solutions_to_population();
      CUOPT_LOG_INFO("Heuristics setup: external solutions added elapsed=%.2f",
                     timer.elapsed_time());
      try {
        run_slope_scaling();
      } catch (const std::exception& e) {
        CUOPT_LOG_INFO("Slope scaling aborted: %s", e.what());
      }
      population.add_external_solutions_to_population();
    };
  }
  generate_solution(timer.remaining_time(), false);
  if (timer.check_time_limit()) {
    population.add_external_solutions_to_population();
    return population.best_feasible();
  }
  if (check_b_b_preemption()) {
    population.add_external_solutions_to_population();
    return population.best_feasible();
  }

  run_fp_alone();
  population.add_external_solutions_to_population();
  return population.best_feasible();
};

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::diversity_step(i_t max_iterations_without_improvement)
{
  bool improved = true;
  while (improved) {
    int k    = max_iterations_without_improvement;
    improved = false;
    while (k-- > 0) {
      if (check_b_b_preemption()) { return; }
      auto new_sol_vector = population.get_external_solutions();
      recombine_and_ls_with_all(new_sol_vector);
      population.adjust_weights_according_to_best_feasible();
      cuopt_assert(population.test_invariant(), "");
      if (population.current_size() < 2) {
        CUOPT_LOG_DEBUG("Population degenerated in diversity step");
        return;
      }
      if (timer.check_time_limit()) return;
      constexpr bool tournament = true;
      auto [sol1, sol2]         = population.get_two_random(tournament);
      cuopt_assert(population.test_invariant(), "");
      auto [lp_offspring, offspring] = recombine_and_local_search(sol1, sol2);
      auto [inserted_pos_1, best_updated_1] =
        population.add_solution(std::move(lp_offspring), "diversity_step_lp_offspring");
      auto [inserted_pos_2, best_updated_2] =
        population.add_solution(std::move(offspring), "diversity_step_offspring");
      if (best_updated_1 || best_updated_2) { recombine_stats.add_best_updated(); }
      cuopt_assert(population.test_invariant(), "");
      if ((inserted_pos_1 != -1 && inserted_pos_1 <= 2) ||
          (inserted_pos_2 != -1 && inserted_pos_2 <= 2)) {
        improved = true;
        recombine_stats.print();
        break;
      }
    }
  }
  recombine_stats.print();
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::set_new_user_bound(f_t new_bound)
{
  stats.set_solution_bound(new_bound);
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::recombine_and_ls_with_all(solution_t<i_t, f_t>& solution,
                                                              bool add_only_feasible)
{
  raft::common::nvtx::range fun_scope("recombine_and_ls_with_all");
  // if (population.population_hash_map.check_skip_solution(solution, 1)) { return; }
  auto population_vector = population.population_to_vector();
  for (auto& curr_sol : population_vector) {
    if (check_integer_equal_on_indices(problem_ptr->integer_indices,
                                       curr_sol.assignment,
                                       solution.assignment,
                                       problem_ptr->tolerances.integrality_tolerance,
                                       problem_ptr->handle_ptr)) {
      CUOPT_LOG_DEBUG("Skipping solution because it is equal to the given solution");
      continue;
    }
    for (const auto recombiner_type : recombiner_t<i_t, f_t>::enabled_recombiners) {
      if (check_b_b_preemption()) { return; }
      if (curr_sol.get_feasible()) {
        auto [offspring, lp_offspring] =
          recombine_and_local_search(curr_sol, solution, recombiner_type);
        if (!add_only_feasible || lp_offspring.get_feasible()) {
          population.add_solution(std::move(lp_offspring), "recombine_all_lp_offspring");
        }
        if (!add_only_feasible || offspring.get_feasible()) {
          population.add_solution(std::move(offspring), "recombine_all_offspring");
        }
        if (timer.check_time_limit()) { return; }
      }
    }
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::recombine_and_ls_with_all(
  std::vector<solution_t<i_t, f_t>>& solutions, bool add_only_feasible)
{
  raft::common::nvtx::range fun_scope("recombine_and_ls_with_all");
  if (solutions.size() > 0) {
    CUOPT_LOG_DEBUG("Running recombiners on B&B solutions with size %lu", solutions.size());
    // add all solutions because time limit might have been consumed and we might have exited before
    for (auto& sol : solutions) {
      cuopt_func_call(sol.test_feasibility(true));
      population.add_solution(std::move(solution_t<i_t, f_t>(sol)), "bnb_solution_import");
    }
    for (auto& sol : solutions) {
      if (timer.check_time_limit()) { return; }
      solution_t<i_t, f_t> ls_solution(sol);
      ls_config_t<i_t, f_t> ls_config;
      run_local_search(ls_solution, population.weights, timer, ls_config);
      if (timer.check_time_limit()) { return; }
      // TODO try if running LP with integers fixed makes it feasible
      if (ls_solution.get_feasible()) {
        CUOPT_LOG_DEBUG("LS searched solution feasible, running recombiners!");
        recombine_and_ls_with_all(ls_solution, add_only_feasible);
      } else {
        CUOPT_LOG_DEBUG("Given solution feasible, running recombiners!");
        recombine_and_ls_with_all(sol, add_only_feasible);
      }
    }
  }
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::check_better_than_both(solution_t<i_t, f_t>& offspring,
                                                           solution_t<i_t, f_t>& sol1,
                                                           solution_t<i_t, f_t>& sol2)
{
  bool better_than_both = false;
  if (sol1.get_feasible() && sol2.get_feasible()) {
    better_than_both = offspring.get_objective() <
                       (std::min(sol1.get_objective(), sol2.get_objective()) - OBJECTIVE_EPSILON);
  } else if (sol1.get_feasible()) {
    better_than_both = offspring.get_objective() < (sol1.get_objective() - OBJECTIVE_EPSILON);
  } else if (sol2.get_feasible()) {
    better_than_both = offspring.get_objective() < (sol2.get_objective() - OBJECTIVE_EPSILON);
  } else {
    better_than_both = offspring.get_feasible();
  }
  if (offspring.get_feasible() && better_than_both) {
    context.settings.benchmark_info_ptr->last_improvement_after_recombination =
      timer.elapsed_time();
  }
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>>
diversity_manager_t<i_t, f_t>::recombine_and_local_search(solution_t<i_t, f_t>& sol1,
                                                          solution_t<i_t, f_t>& sol2,
                                                          recombiner_enum_t recombiner_type)
{
  raft::common::nvtx::range fun_scope("recombine_and_local_search");
  CUOPT_LOG_DEBUG("Recombining sol cost:feas %f : %d and %f : %d",
                  sol1.get_quality(population.weights),
                  sol1.get_feasible(),
                  sol2.get_quality(population.weights),
                  sol2.get_feasible());
  double best_objective_of_parents  = std::min(sol1.get_objective(), sol2.get_objective());
  bool at_least_one_parent_feasible = sol1.get_feasible() || sol2.get_feasible();
  // randomly choose among 3 recombiners
  auto [offspring, success] = recombine(sol1, sol2, recombiner_type);
  if (!success) {
    // add the attempt
    mab_recombiner.add_mab_reward(mab_recombiner.last_chosen_option,
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::max(),
                                  recombiner_work_normalized_reward_t(0.0));
    return std::make_pair(solution_t<i_t, f_t>(sol1), solution_t<i_t, f_t>(sol2));
  }
  cuopt_assert(population.test_invariant(), "");
  cuopt_func_call(offspring.test_variable_bounds(false));
  CUOPT_LOG_DEBUG("Recombiner offspring sol cost:feas %f : %d",
                  offspring.get_quality(population.weights),
                  offspring.get_feasible());
  cuopt_assert(offspring.test_number_all_integer(), "All must be integers before LS");
  bool feasibility_before = offspring.get_feasible();
  ls_config_t<i_t, f_t> ls_config;
  ls_config.best_objective_of_parents    = best_objective_of_parents;
  ls_config.at_least_one_parent_feasible = at_least_one_parent_feasible;
  success = this->run_local_search(offspring, population.weights, timer, ls_config);
  if (!success) {
    // add the attempt
    mab_recombiner.add_mab_reward(mab_recombiner.last_chosen_option,
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::lowest(),
                                  std::numeric_limits<double>::max(),
                                  recombiner_work_normalized_reward_t(0.0));
    return std::make_pair(solution_t<i_t, f_t>(sol1), solution_t<i_t, f_t>(sol2));
  }
  cuopt_assert(offspring.test_number_all_integer(), "All must be integers after LS");
  cuopt_assert(population.test_invariant(), "");
  offspring.compute_feasibility();
  CUOPT_LOG_DEBUG("After LS offspring sol cost:feas %f : %d",
                  offspring.get_quality(population.weights),
                  offspring.get_feasible());
  cuopt_assert(population.test_invariant(), "");
  // run LP with the vars
  solution_t<i_t, f_t> lp_offspring(offspring);
  cuopt_assert(population.test_invariant(), "");
  cuopt_assert(lp_offspring.test_number_all_integer(), "All must be integers before LP");
  f_t lp_run_time = offspring.get_feasible() ? diversity_config.lp_run_time_if_feasible
                                             : diversity_config.lp_run_time_if_infeasible;
  lp_run_time     = std::min(lp_run_time, timer.remaining_time());
  relaxed_lp_settings_t lp_settings;
  lp_settings.time_limit              = lp_run_time;
  lp_settings.tolerance               = context.settings.tolerances.absolute_tolerance;
  lp_settings.return_first_feasible   = false;
  lp_settings.save_state              = true;
  lp_settings.per_constraint_residual = true;
  run_lp_with_vars_fixed(*lp_offspring.problem_ptr,
                         lp_offspring,
                         lp_offspring.problem_ptr->integer_indices,
                         lp_settings,
                         &ls.constraint_prop.bounds_update,
                         true /* check fixed assignment is feasible */,
                         true /* use integer fixed problem */);
  cuopt_assert(population.test_invariant(), "");
  cuopt_assert(lp_offspring.test_number_all_integer(), "All must be integers after LP");
  f_t lp_qual = lp_offspring.get_quality(population.weights);
  CUOPT_LOG_DEBUG("After LP offspring sol cost:feas %f : %d", lp_qual, lp_offspring.get_feasible());
  f_t offspring_qual = std::min(offspring.get_quality(population.weights), lp_qual);
  recombine_stats.update_improve_stats(
    offspring_qual, sol1.get_quality(population.weights), sol2.get_quality(population.weights));
  f_t best_quality_of_parents =
    std::min(sol1.get_quality(population.weights), sol2.get_quality(population.weights));
  mab_recombiner.add_mab_reward(
    mab_recombiner.last_chosen_option,
    best_quality_of_parents,
    population.best().get_quality(population.weights),
    offspring_qual,
    recombiner_work_normalized_reward_t(recombine_stats.get_last_recombiner_time()));
  mab_ls.add_mab_reward(mab_ls_config_t<i_t, f_t>::last_ls_mab_option,
                        best_quality_of_parents,
                        population.best_feasible().get_quality(population.weights),
                        offspring_qual,
                        ls_work_normalized_reward_t(mab_ls_config_t<i_t, f_t>::last_lm_config));
  if (context.settings.benchmark_info_ptr != nullptr) {
    check_better_than_both(offspring, sol1, sol2);
    check_better_than_both(lp_offspring, sol1, sol2);
  }
  return std::make_pair(std::move(offspring), std::move(lp_offspring));
}

template <typename i_t, typename f_t>
std::pair<solution_t<i_t, f_t>, bool> diversity_manager_t<i_t, f_t>::recombine(
  solution_t<i_t, f_t>& a, solution_t<i_t, f_t>& b, recombiner_enum_t recombiner_type)
{
  recombiner_enum_t recombiner;
  i_t selected_index = -1;
  if (run_only_ls_recombiner) {
    recombiner = recombiner_enum_t::LINE_SEGMENT;
  } else if (run_only_bp_recombiner) {
    recombiner = recombiner_enum_t::BOUND_PROP;
  } else if (run_only_fp_recombiner) {
    recombiner = recombiner_enum_t::FP;
  } else if (run_only_sub_mip_recombiner) {
    recombiner = recombiner_enum_t::SUB_MIP;
  } else {
    // only run the given recombiner unless it is defult
    if (recombiner_type == recombiner_enum_t::SIZE) {
      selected_index = mab_recombiner.select_mab_option();
      recombiner     = recombiner_t<i_t, f_t>::enabled_recombiners[selected_index];
    } else {
      recombiner = recombiner_type;
      auto it    = std::find(recombiner_t<i_t, f_t>::enabled_recombiners.begin(),
                          recombiner_t<i_t, f_t>::enabled_recombiners.end(),
                          recombiner_type);
      selected_index =
        static_cast<i_t>(std::distance(recombiner_t<i_t, f_t>::enabled_recombiners.begin(), it));
      if (it == recombiner_t<i_t, f_t>::enabled_recombiners.end()) {
        CUOPT_LOG_DEBUG("Recombiner not enabled; falling back to index 0");
        selected_index = 0;
      }
    }
  }
  mab_recombiner.set_last_chosen_option(selected_index);
  recombine_stats.add_attempt((recombiner_enum_t)recombiner);
  recombine_stats.start_recombiner_time();
  // Refactored code using a switch statement
  switch (recombiner) {
    case recombiner_enum_t::BOUND_PROP: {
      auto [sol, success] = bound_prop_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::FP: {
      auto [sol, success] = fp_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::LINE_SEGMENT: {
      auto [sol, success] = line_segment_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::SUB_MIP: {
      auto [sol, success] = sub_mip_recombiner.recombine(a, b, population.weights);
      recombine_stats.stop_recombiner_time();
      if (success) { recombine_stats.add_success(); }
      return std::make_pair(sol, success);
    }
    case recombiner_enum_t::SIZE: {
      CUOPT_LOG_ERROR("Invalid or unhandled recombiner type: %d", recombiner);
      return std::make_pair(solution_t<i_t, f_t>(a), false);
    }
  }
  CUOPT_LOG_ERROR("Invalid or unhandled recombiner type: %d", recombiner);
  return std::make_pair(solution_t<i_t, f_t>(a), false);
}

template <typename i_t, typename f_t>
void diversity_manager_t<i_t, f_t>::set_simplex_solution(const std::vector<f_t>& solution,
                                                         const std::vector<f_t>& dual_solution,
                                                         f_t objective)
{
  CUOPT_LOG_DEBUG("Setting simplex solution with objective %f", objective);
  std::lock_guard<std::mutex> lock(relaxed_solution_mutex);
  global_concurrent_halt = 1;
  cuopt_assert(lp_optimal_solution.size() == solution.size(), "Assignment size mismatch");
  cuopt_assert(problem_ptr->n_constraints == dual_solution.size(), "Dual assignment size mismatch");
  staged_simplex_solution      = solution;
  staged_simplex_dual_solution = dual_solution;
  staged_simplex_objective     = objective;
  simplex_solution_exists.store(true, std::memory_order_release);
  CUOPT_LOG_DEBUG("Staged simplex solution and requested concurrent halt");
}

#if MIP_INSTANTIATE_FLOAT
template class diversity_manager_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class diversity_manager_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
