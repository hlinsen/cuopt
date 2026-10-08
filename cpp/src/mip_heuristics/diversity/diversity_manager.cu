/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "cuda_profiler_api.h"
#include "diversity_manager.cuh"

#include <mip_heuristics/feasibility_jump/early_cpufj.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/presolve/third_party_presolve.hpp>

#include <mip_heuristics/presolve/block_bve.cuh>
#include <mip_heuristics/presolve/conflict_graph/clique_table.cuh>
#include <mip_heuristics/presolve/probing_cache.cuh>
#include <mip_heuristics/presolve/trivial_presolve.cuh>
#include <mip_heuristics/problem/problem_helpers.cuh>

#include <cuopt/mathematical_optimization/solve.hpp>
#include <pdlp/root_lp_snapshot.cuh>
#include <pdlp/solve.cuh>
#include <mip_heuristics/relaxed_lp/relaxed_lp.cuh>
#include "fix_propagate_host.hpp"

#include <utilities/copy_helpers.hpp>
#include <utilities/scope_guard.hpp>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

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
    original_rows == cuopt::host_copy(root.constraint_lower_bounds, root.handle_ptr->get_stream()) &&
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
  auto& lp_state = fixed_problem.lp_state;
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
    auto resp = get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, lp_settings);
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
    if (feas) { return true; }
    if (status == pdlp_termination_status_t::PrimalInfeasible ||
        status == pdlp_termination_status_t::DualInfeasible ||
        status == pdlp_termination_status_t::NumericalError) {
      return false;
    }
  }
  return false;
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::polish_continuous(solution_t<i_t, f_t>& in_sol, f_t budget)
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
  const double t_start = timer.elapsed_time();
  auto [fixed_problem, fixed_assignment, variable_map] =
    sol.fix_variables(problem_ptr->integer_indices);
  fixed_problem.check_problem_representation(true);
  rmm::device_uvector<f_t> feasible_start(fixed_assignment, stream);

  // stage 1: cost-aware fixed LP (same solver path as the root LP)
  auto snapshot = make_root_lp_snapshot(fixed_problem);
  pdlp_solver_settings_t<i_t, f_t> lp_settings{};
  lp_settings.time_limit = ptimer.remaining_time() * 0.7;
  lp_settings.method     = context.settings.method;
  std::atomic<int> local_halt{0};
  lp_settings.concurrent_halt = &local_halt;
  lp_settings.inside_mip      = false;
  lp_settings.num_gpus        = context.settings.num_gpus;
  CUOPT_LOG_INFO("Polish start: obj=%g vars=%d cstrs=%d budget=%.1fs lp_budget=%.1fs elapsed=%.2f",
                 obj0,
                 fixed_problem.n_variables,
                 fixed_problem.n_constraints,
                 budget,
                 lp_settings.time_limit,
                 timer.elapsed_time());
  auto lp_result = solve_lp(snapshot, lp_settings);
  const auto& lp_info = lp_result.get_additional_termination_information();
  CUOPT_LOG_INFO(
    "Polish LP: status=%s method=%s lp_obj=%g iters=%d l2_primal_res=%g time=%.2fs elapsed=%.2f",
    lp_result.get_termination_status_string().c_str(),
    method_to_string(lp_info.solved_by).c_str(),
    (double)lp_result.get_objective_value(),
    lp_info.number_of_steps_taken,
    lp_info.l2_primal_residual,
    timer.elapsed_time() - t_start,
    timer.elapsed_time());
  if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables) {
    CUOPT_LOG_INFO("Polish LP returned no usable primal");
    return false;
  }
  raft::copy(fixed_assignment.data(),
             lp_result.get_primal_solution().data(),
             fixed_assignment.size(),
             stream);
  clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
  rmm::device_uvector<f_t> cost_aware(fixed_assignment, stream);
  sol.unfix_variables(fixed_assignment, variable_map);
  bool feas  = sol.get_feasible();
  f_t obj_lp = sol.get_user_objective();
  CUOPT_LOG_INFO("Polish LP point: feasible=%d obj=%g max_cstr_viol=%g excess=%g elapsed=%.2f",
                 (int)feas,
                 obj_lp,
                 sol.compute_max_constraint_violation(),
                 sol.get_total_excess(),
                 timer.elapsed_time());
  if (feas && obj_lp < obj0) {
    population.add_solution(std::move(sol), "polish_lp");
    return true;
  }

  // stage 2: objective-free repair warm-started from the cost-aware point
  thrust::fill(handle_ptr->get_thrust_policy(),
               fixed_problem.objective_coefficients.begin(),
               fixed_problem.objective_coefficients.end(),
               f_t(0));
  fixed_problem.presolve_data.objective_offset = 0;
  auto& lp_state = fixed_problem.lp_state;
  lp_state.resize(fixed_problem, stream);
  thrust::fill(handle_ptr->get_thrust_policy(), lp_state.prev_dual.begin(), lp_state.prev_dual.end(), f_t(0));
  raft::copy(fixed_assignment.data(), cost_aware.data(), cost_aware.size(), stream);
  relaxed_lp_settings_t r_settings;
  r_settings.tolerance               = problem_ptr->tolerances.absolute_tolerance;
  r_settings.return_first_feasible   = true;
  r_settings.save_state              = true;
  r_settings.check_infeasibility     = false;
  r_settings.per_constraint_residual = true;
  r_settings.has_initial_primal      = true;
  int chunk_id = 0;
  while (!ptimer.check_time_limit() && !timer.check_time_limit()) {
    if (check_b_b_preemption()) { return false; }
    r_settings.time_limit = std::min<double>(20., ptimer.remaining_time());
    if (r_settings.time_limit < 1.) { break; }
    auto resp = get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, r_settings);
    const auto& info = resp.get_additional_termination_information();
    sol.unfix_variables(fixed_assignment, variable_map);
    feas    = sol.get_feasible();
    f_t obj = sol.get_user_objective();
    CUOPT_LOG_INFO(
      "Polish repair chunk %d: status=%s iters=%d feasible=%d obj=%g max_cstr_viol=%g elapsed=%.2f",
      chunk_id,
      resp.get_termination_status_string().c_str(),
      info.number_of_steps_taken,
      (int)feas,
      obj,
      sol.compute_max_constraint_violation(),
      timer.elapsed_time());
    chunk_id++;
    if (feas) {
      if (obj < obj0) {
        population.add_solution(std::move(sol), "polish_repair");
        return true;
      }
      return false;
    }
  }
  return false;
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
  early_fpc_done       = true;
  population.timer     = timer;
  fpc_external_publish = true;
  CUOPT_LOG_INFO("Early fix-and-propagate start: elapsed=%.2f", timer.elapsed_time());
  run_fix_propagate_complete(false, 25.);
  fpc_external_publish = false;
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_fix_propagate_complete(bool use_lp, f_t max_budget)
{
  raft::common::nvtx::range fun_scope("run_fix_propagate_complete");
  auto* handle_ptr     = problem_ptr->handle_ptr;
  auto stream          = handle_ptr->get_stream();
  const f_t budget     = std::min<f_t>(timer.remaining_time() * 0.8, max_budget);
  timer_t fpc_timer(budget);
  const f_t int_tol    = problem_ptr->tolerances.integrality_tolerance;
  const double t_build = timer.elapsed_time();

  // host copies of the model and the LP optimum
  const i_t n_vars  = problem_ptr->n_variables;
  const i_t n_cstrs = problem_ptr->n_constraints;
  auto h_offsets    = cuopt::host_copy(problem_ptr->offsets, stream);
  auto h_cols       = cuopt::host_copy(problem_ptr->variables, stream);
  auto h_vals       = cuopt::host_copy(problem_ptr->coefficients, stream);
  auto h_clb        = cuopt::host_copy(problem_ptr->constraint_lower_bounds, stream);
  auto h_cub        = cuopt::host_copy(problem_ptr->constraint_upper_bounds, stream);
  auto h_int        = cuopt::host_copy(problem_ptr->integer_indices, stream);
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
  fp_host.build(n_vars, n_cstrs, h_offsets, h_cols, h_vals, h_clb, h_cub, h_vlb, h_vub, h_int);
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

  // processing order: most decided (closest to integral) first
  std::vector<i_t> order(h_int.begin(), h_int.end());
  std::stable_sort(order.begin(), order.end(), [&](i_t a, i_t b) {
    f_t fa = std::abs(h_lp[a] - std::floor(h_lp[a]) - 0.5);
    f_t fb = std::abs(h_lp[b] - std::floor(h_lp[b]) - 0.5);
    return fa > fb;
  });

  // with the LP: nearest / up-biased / down-biased; without: upper bounds / lower bounds
  const int n_attempts = use_lp ? 3 : 2;
  for (int attempt = 0; attempt < n_attempts; ++attempt) {
    if (fpc_timer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) {
      break;
    }
    // value preference: 0 = LP nearest, 1 = up-biased (any positive LP value -> up),
    // 2 = down-biased (only LP values at the upper end -> up)
    std::vector<f_t> pref(n_vars, 0.);
    for (auto v : h_int) {
      f_t x  = h_lp[v];
      f_t fl = std::floor(x + int_tol);
      f_t fr = x - fl;
      if (!use_lp) {
        pref[v] = attempt == 0 ? h_vub[v] : h_vlb[v];
      } else if (attempt == 0) {
        pref[v] = std::round(x);
      } else if (attempt == 1) {
        pref[v] = fr > int_tol ? fl + 1 : fl;
      } else {
        pref[v] = fr >= 1. - int_tol ? fl + 1 : fl;
      }
    }
    const double t_dive = timer.elapsed_time();
    i_t n_flipped = 0, n_implied = 0;
    fp_host.work      = 0;
    i_t n_conflicts   = fp_host.dive(order, pref, n_flipped, n_implied);
    i_t n_dead        = 0;
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
      "FPC attempt %d dive: time=%.3fs conflicts=%d dead_rows=%d flipped=%d implied_off_pref=%d "
      "ones=%d differs_from_lp_nearest=%d work=%ld feasible=%d max_cstr_viol=%g elapsed=%.2f",
      attempt,
      timer.elapsed_time() - t_dive,
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
    if (!feas) { feas = fpc_complete_continuous(sol, fpc_timer, attempt, !use_lp); }
    if (feas) {
      CUOPT_LOG_INFO("FPC attempt %d found feasible solution obj=%g elapsed=%.2f",
                     attempt,
                     sol.get_user_objective(),
                     timer.elapsed_time());
      if (fpc_external_publish) {
        population.add_external_solution(
          sol.get_host_assignment(), sol.get_objective(), solution_origin_t::FIX_PROPAGATE);
      } else {
        solution_t<i_t, f_t> to_polish(sol);
        population.add_solution(std::move(sol),
                                use_lp ? "fix_propagate_complete" : "fix_propagate_complete_nolp");
        if (use_lp) { polish_continuous(to_polish, 300.); }
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
  const bool has_papilo   = problem_ptr->has_papilo_presolve_data();
  const i_t papilo_orig_n = problem_ptr->get_papilo_original_num_variables();
  for (size_t sol_idx = 0; sol_idx < context.settings.initial_solutions.size(); ++sol_idx) {
    if (timer.check_time_limit()) { break; }
    const auto& init_sol = context.settings.initial_solutions[sol_idx];
    solution_t<i_t, f_t> sol(*problem_ptr);
    rmm::device_uvector<f_t> init_sol_assignment(*init_sol, sol.handle_ptr->get_stream());

    if (has_papilo) {
      if ((i_t)init_sol_assignment.size() != papilo_orig_n) {
        CUOPT_LOG_ERROR(
          "add the provided initial solution! Initial solution %zu has %zu vars, "
          "expected %d; skipping",
          sol_idx,
          init_sol_assignment.size(),
          papilo_orig_n);
        continue;
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
  }
}

template <typename i_t, typename f_t>
bool diversity_manager_t<i_t, f_t>::run_presolve(f_t time_limit, timer_t global_timer)
{
  raft::common::nvtx::range fun_scope("run_presolve");
  CUOPT_LOG_INFO("\nRunning cuOpt presolve");
  timer_t presolve_timer(time_limit);

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
    ls.constraint_prop.bounds_update.settings.num_tasks =
      std::max(1, omp_get_num_threads() - 1 - held_by_cpufj);
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
    CUOPT_LOG_DEBUG(
      "PRESOLVE_PROBING_WALL wall=%.3f",
      std::chrono::duration<double>(std::chrono::steady_clock::now() - probing_t0).count());
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
  ls.fj.solve(solution);
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

  population.timer        = timer;
  const f_t time_limit    = timer.remaining_time();
  const auto& hp          = context.settings.heuristic_params;
  const f_t lp_time_limit = std::min(hp.root_lp_max_time, time_limit * hp.root_lp_time_ratio);
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
  population.initialize_population();
  population.allocate_solutions();
  add_user_given_solutions(initial_sol_vector);
  population.add_solutions_from_vec(std::move(initial_sol_vector), "initial_solutions");
  if (check_b_b_preemption()) { return population.best_feasible(); }
  // Run CPUFJ early to find quick initial solutions
  ls_cpufj_raii_guard_t ls_cpufj_raii_guard(ls);  // RAII to stop cpufj threads on solve stop
  ls.start_cpufj_scratch_threads(population);

  if (check_b_b_preemption()) { return population.best_feasible(); }
  CUOPT_LOG_INFO("Heuristics start (before root LP): elapsed=%.2f", timer.elapsed_time());
  // LP-free fix-and-propagate attempt before the root LP (bounded, objective-free completion)
  if (!fj_only_run && !early_fpc_done && !simplex_solution_exists.load() &&
      !check_b_b_preemption()) {
    run_fix_propagate_complete(false, 25.);
    population.add_external_solutions_to_population();
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
    pdlp_settings.time_limit              = lp_time_limit;
    pdlp_settings.first_primal_feasible   = false;
    pdlp_settings.concurrent_halt         = &global_concurrent_halt;
    pdlp_settings.method                  = context.settings.method;
    pdlp_settings.inside_mip              = false;
    pdlp_settings.num_gpus                = context.settings.num_gpus;
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

    // The concurrent root LP can fail to produce a usable solution -- e.g. the barrier
    // hits a numerical error on an infeasible problem and PDLP returns NumericalError
    // with empty primal/dual. In that case we must not copy or hand off the empty
    // result (copying n elements from an empty buffer throws), and we must still
    // release B&B's root-relaxation wait so it proceeds with its own dual-simplex root
    // instead of spinning forever.
    const bool root_lp_usable =
      lp_result.get_termination_status() != pdlp_termination_status_t::NumericalError &&
      lp_result.get_primal_solution().size() == lp_optimal_solution.size() &&
      lp_result.get_dual_solution().size() == lp_dual_optimal_solution.size();

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
      problem_ptr->set_root_relaxation_solution_callback(
        host_primal, host_dual, host_reduced_costs, solver_obj, user_obj, iterations, method);
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
    ls.start_cpufj_lptopt_scratch_threads(population);
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

  if (ls.lp_optimal_exists && !check_b_b_preemption() && !timer.check_time_limit()) {
    run_fix_propagate_complete(true, 700.);
    population.add_external_solutions_to_population();
    if (timer.check_time_limit() || check_b_b_preemption()) { return population.best_feasible(); }
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
      auto [lp_offspring, offspring]        = recombine_and_local_search(sol1, sol2);
      auto [inserted_pos_1, best_updated_1] = population.add_solution(std::move(lp_offspring), "diversity_step_lp_offspring");
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
