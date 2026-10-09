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
#include <random>
#include <string>

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
    if (feas) {
      f_t max_excess = 0;
      if (strict_point_ok(sol.get_host_assignment(), max_excess)) { return true; }
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
  const size_t k_end =
    last_stage < 0 ? n_stages : std::min(n_stages, (size_t)last_stage + 1);
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
    if (ptimer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption()) {
      break;
    }
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
    const bool warm = warm_primal.size() == (size_t)fixed_problem.n_variables &&
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
      raft::copy(warm_primal.data(),
                 lp_result.get_primal_solution().data(),
                 warm_primal.size(),
                 stream);
      warm_dual.resize(lp_result.get_dual_solution().size(), stream);
      raft::copy(
        warm_dual.data(), lp_result.get_dual_solution().data(), warm_dual.size(), stream);
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
      if (!strict_point_ok(sol.get_host_assignment(), max_excess)) {
        CUOPT_LOG_INFO("Polish stage %d LP point: strict check failed excess=%g", (int)k, max_excess);
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
    thrust::fill(
      handle_ptr->get_thrust_policy(), lp_state.prev_dual.begin(), lp_state.prev_dual.end(), f_t(0));
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
      auto resp        = get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, r_settings);
      const auto& info = resp.get_additional_termination_information();
      sol.unfix_variables(fixed_assignment, variable_map);
      feas           = sol.get_feasible();
      f_t obj        = sol.get_user_objective();
      f_t max_excess = 0;
      if (feas && !strict_point_ok(sol.get_host_assignment(), max_excess)) {
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
                                                  f_t repair_below)
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
    auto snapshot = make_root_lp_snapshot(fixed_problem);
    pdlp_solver_settings_t<i_t, f_t> lp_settings{};
    lp_settings.set_optimality_tolerance(tol);
    lp_settings.time_limit = std::max<double>(1., tlimit);
    lp_settings.method     = method_t::PDLP;
    if (warm) { lp_settings.presolver = presolver_t::None; }
    std::atomic<int> local_halt{0};
    lp_settings.concurrent_halt = &local_halt;
    lp_settings.inside_mip      = false;
    lp_settings.num_gpus        = context.settings.num_gpus;
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
    auto lp_result      = solve_lp(snapshot, lp_settings);
    const auto& lp_info = lp_result.get_additional_termination_information();
    res.iters           = lp_info.number_of_steps_taken;
    res.l2_primal_res   = lp_info.l2_primal_residual;
    res.optimal = lp_result.get_termination_status() == pdlp_termination_status_t::Optimal;
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
  if (repair && res.usable && res.lp_obj < repair_below) {
    // phase B: the PID point's l2 primal residual stalls ~50-100 (repair +2%); warm-started
    // pushes with a frozen large primal weight drive it to ~1e-3 at a nearly unchanged objective
    // (the ranking objective above stays the phase-A one)
    const double t_push = timer.elapsed_time();
    struct push_t {
      double weight;
      double cap;
    };
    const push_t pushes[] = {{1e4, 6.}, {1e5, 5.}};
    auto cur_p            = cuopt::device_copy(res.primal, stream);
    auto cur_d            = cuopt::device_copy(res.dual, stream);
    bool pushed           = false;
    for (const auto& ps : pushes) {
      if (timer.check_time_limit() || check_b_b_preemption() || timer.remaining_time() < 30.) {
        break;
      }
      auto snapshot = make_root_lp_snapshot(fixed_problem);
      pdlp_solver_settings_t<i_t, f_t> lp_settings{};
      lp_settings.set_optimality_tolerance(1e-7);
      lp_settings.time_limit = ps.cap;
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
      auto lp_result = solve_lp(snapshot, lp_settings);
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
    bool feas = sol.get_feasible();
    if (feas) {
      f_t max_excess = 0;
      feas           = strict_point_ok(sol.get_host_assignment(), max_excess);
    }
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
        if (feas) {
          f_t max_excess = 0;
          if (!strict_point_ok(sol.get_host_assignment(), max_excess)) {
            CUOPT_LOG_INFO("Pattern repair: strict check failed excess=%g", max_excess);
            feas = false;
            r_settings.tolerance *= 0.1;
          }
        }
      }
    }
    if (feas) {
      res.repaired_obj        = sol.get_user_objective();
      res.repaired_assignment = sol.get_host_assignment();
      const f_t best_pub      = population.is_feasible()
                                  ? population.best_feasible().get_user_objective()
                                  : std::numeric_limits<f_t>::infinity();
      if (res.repaired_obj < best_pub) {
        f_t max_excess = 0;
        if (strict_point_ok(res.repaired_assignment, max_excess)) {
          res.published = true;
          population.add_solution(solution_t<i_t, f_t>(sol), origin);
        }
      }
    }
  }
  res.time = timer.elapsed_time() - t0;
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
void diversity_manager_t<i_t, f_t>::dual_flip_loop()
{
  raft::common::nvtx::range fun_scope("dual_flip_loop");
  if (!flip_state.valid || problem_ptr->n_integer_vars == 0 ||
      problem_ptr->n_integer_vars == problem_ptr->n_variables) {
    return;
  }
  auto* handle_ptr     = problem_ptr->handle_ptr;
  auto stream          = handle_ptr->get_stream();
  const i_t n_vars     = problem_ptr->n_variables;
  const i_t n_cstrs    = problem_ptr->n_constraints;
  const double t_start = timer.elapsed_time();
  if ((i_t)flip_state.assignment.size() != n_vars || (i_t)flip_state.dual.size() != n_cstrs) {
    return;
  }
  // host model
  auto h_offsets      = cuopt::host_copy(problem_ptr->offsets, stream);
  auto h_cols         = cuopt::host_copy(problem_ptr->variables, stream);
  auto h_vals         = cuopt::host_copy(problem_ptr->coefficients, stream);
  auto h_clb          = cuopt::host_copy(problem_ptr->constraint_lower_bounds, stream);
  auto h_cub          = cuopt::host_copy(problem_ptr->constraint_upper_bounds, stream);
  auto h_int          = cuopt::host_copy(problem_ptr->integer_indices, stream);
  auto h_obj          = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
  auto [h_vlb, h_vub] = cuopt::extract_host_bounds<f_t>(problem_ptr->variable_bounds, handle_ptr);
  host_fix_propagate_t<i_t, f_t> fp_host;
  fp_host.abs_tol = problem_ptr->tolerances.absolute_tolerance;
  fp_host.build(n_vars, n_cstrs, h_offsets, h_cols, h_vals, h_clb, h_cub, h_vlb, h_vub, h_int);
  std::vector<char> is_bin(n_vars, 0);
  std::vector<i_t> bins;
  for (auto v : h_int) {
    if (h_vlb[v] == 0 && h_vub[v] == 1) {
      is_bin[v] = 1;
      bins.push_back(v);
    }
  }
  if (bins.empty()) { return; }

  const f_t abs_tol = problem_ptr->tolerances.absolute_tolerance;

  std::vector<f_t> cur_assign = flip_state.assignment;
  std::vector<f_t> cur_primal = flip_state.primal;
  std::vector<f_t> cur_dual   = flip_state.dual;
  f_t best_pub                = population.is_feasible()
                                  ? population.best_feasible().get_user_objective()
                                  : std::numeric_limits<f_t>::infinity();

  struct eval_result_t {
    bool usable{false};
    bool optimal{false};
    f_t lp_obj{0};
    f_t repaired_obj{std::numeric_limits<f_t>::infinity()};
    bool published{false};
    int iters{0};
    double time{0};
    std::vector<f_t> primal, dual;
  };
  // solve the fixed LP of a pattern (real objective, PDLP warm-started from the current
  // primal/dual), optionally repair the point to strict feasibility and publish it
  auto evaluate = [&](const std::vector<f_t>& assign, f_t tol, f_t tlimit, bool repair) {
    eval_result_t res;
    const double t0 = timer.elapsed_time();
    solution_t<i_t, f_t> sol(*problem_ptr);
    sol.copy_new_assignment(assign);
    auto [fixed_problem, fixed_assignment, variable_map] =
      sol.fix_variables(problem_ptr->integer_indices);
    fixed_problem.check_problem_representation(true);
    if ((size_t)fixed_problem.n_variables != cur_primal.size() ||
        fixed_problem.n_constraints != n_cstrs) {
      return res;
    }
    {
      auto snapshot = make_root_lp_snapshot(fixed_problem);
      pdlp_solver_settings_t<i_t, f_t> lp_settings{};
      lp_settings.set_optimality_tolerance(tol);
      lp_settings.time_limit = std::max<double>(1., tlimit);
      lp_settings.method     = method_t::PDLP;
      lp_settings.presolver  = presolver_t::None;
      std::atomic<int> local_halt{0};
      lp_settings.concurrent_halt = &local_halt;
      lp_settings.inside_mip      = false;
      lp_settings.num_gpus        = context.settings.num_gpus;
      // the snapshot carries no variable types; the initial-solution check reads them
      std::vector<var_t> h_types(fixed_problem.n_variables, var_t::CONTINUOUS);
      snapshot.set_variable_types(h_types.data(), fixed_problem.n_variables);
      // warm-start primal must lie within the variable bounds
      raft::copy(fixed_assignment.data(), cur_primal.data(), cur_primal.size(), stream);
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(
        fixed_assignment.data(), (i_t)fixed_assignment.size(), stream);
      lp_settings.set_initial_dual_solution(cur_dual.data(), (i_t)cur_dual.size(), stream);
      auto lp_result      = solve_lp(snapshot, lp_settings);
      const auto& lp_info = lp_result.get_additional_termination_information();
      res.iters           = lp_info.number_of_steps_taken;
      res.optimal = lp_result.get_termination_status() == pdlp_termination_status_t::Optimal;
      // the fixed problem's objective excludes the fixed integers' cost; add it back so that
      // patterns are compared on their full objective
      f_t int_cost = 0;
      for (auto v : h_int) {
        int_cost += h_obj[v] * assign[v];
      }
      res.lp_obj = lp_result.get_objective_value() +
                   problem_ptr->presolve_data.objective_scaling_factor * int_cost;
      if (lp_result.get_primal_solution().size() != (size_t)fixed_problem.n_variables ||
          lp_result.get_dual_solution().size() != (size_t)n_cstrs) {
        res.time = timer.elapsed_time() - t0;
        return res;
      }
      res.usable = std::isfinite(res.lp_obj);
      res.primal = cuopt::host_copy(lp_result.get_primal_solution(), stream);
      res.dual   = cuopt::host_copy(lp_result.get_dual_solution(), stream);
      raft::copy(fixed_assignment.data(),
                 lp_result.get_primal_solution().data(),
                 fixed_assignment.size(),
                 stream);
    }
    if (repair && res.usable) {
      clamp_within_var_bounds(fixed_assignment, &fixed_problem, handle_ptr);
      sol.unfix_variables(fixed_assignment, variable_map);
      bool feas = sol.get_feasible();
      if (feas) {
        f_t max_excess = 0;
        feas           = strict_point_ok(sol.get_host_assignment(), max_excess);
      }
      if (!feas) {
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
        r_settings.tolerance               = 0.2 * abs_tol;
        r_settings.return_first_feasible   = true;
        r_settings.save_state              = true;
        r_settings.check_infeasibility     = false;
        r_settings.per_constraint_residual = true;
        r_settings.has_initial_primal      = true;
        for (int chunk_id = 0; chunk_id < 3 && !feas; ++chunk_id) {
          if (timer.check_time_limit() || check_b_b_preemption()) { break; }
          r_settings.time_limit = std::min<double>(15., timer.remaining_time() - 1.);
          if (r_settings.time_limit < 1.) { break; }
          get_relaxed_lp_solution(fixed_problem, fixed_assignment, lp_state, r_settings);
          sol.unfix_variables(fixed_assignment, variable_map);
          feas = sol.get_feasible();
          if (feas) {
            f_t max_excess = 0;
            if (!strict_point_ok(sol.get_host_assignment(), max_excess)) {
              CUOPT_LOG_INFO("Dual flip repair: strict check failed excess=%g", max_excess);
              feas = false;
              r_settings.tolerance *= 0.1;
            }
          }
        }
      }
      if (feas) {
        res.repaired_obj = sol.get_user_objective();
        if (res.repaired_obj < best_pub) {
          auto h_x       = sol.get_host_assignment();
          f_t max_excess = 0;
          if (strict_point_ok(h_x, max_excess)) {
            best_pub      = res.repaired_obj;
            res.published = true;
            population.add_solution(solution_t<i_t, f_t>(sol), "dual_flip");
          } else {
            CUOPT_LOG_INFO("Dual flip: repaired point rejected by strict check excess=%g",
                           max_excess);
          }
        }
      }
    }
    res.time = timer.elapsed_time() - t0;
    return res;
  };

  // consistent baseline: the current pattern at the screening tolerance
  constexpr f_t screen_tol = 1e-4;
  if (timer.remaining_time() < 10. || check_b_b_preemption()) { return; }
  auto base = evaluate(cur_assign, screen_tol, std::min<f_t>(20., timer.remaining_time() - 6.), false);
  if (!base.usable) {
    CUOPT_LOG_INFO("Dual flip: baseline evaluation failed elapsed=%.2f", timer.elapsed_time());
    return;
  }
  f_t cur_lp = base.lp_obj;
  cur_primal = std::move(base.primal);
  cur_dual   = std::move(base.dual);
  f_t last_tight_lp      = cur_lp;
  double last_tight_time = timer.elapsed_time();
  CUOPT_LOG_INFO(
    "Dual flip start: bins=%zu tight_lp=%g screen_lp=%g (iters=%d %.2fs) best_pub=%g elapsed=%.2f",
    bins.size(),
    flip_state.lp_obj,
    cur_lp,
    base.iters,
    base.time,
    best_pub,
    timer.elapsed_time());

  std::vector<f_t> red(n_vars, 0);
  std::vector<int> tabu_until(n_vars, 0);
  std::vector<f_t> pref(n_vars, 0);
  int k        = 8;
  int n_iter   = 0;
  int n_accept = 0;
  constexpr int tabu_len  = 40;
  constexpr int k_max     = 1024;
  constexpr double margin = 6.;
  auto run_tight = [&]() {
    const f_t tb = std::min<f_t>(90., timer.remaining_time() - margin - 5.);
    auto tight   = evaluate(cur_assign, 1e-7, tb, true);
    CUOPT_LOG_INFO(
      "Dual flip tight: lp=%g (screen %g) optimal=%d iters=%d repaired=%g published=%d "
      "time=%.2fs best_pub=%g elapsed=%.2f",
      tight.lp_obj,
      cur_lp,
      (int)tight.optimal,
      tight.iters,
      tight.repaired_obj,
      (int)tight.published,
      tight.time,
      best_pub,
      timer.elapsed_time());
    last_tight_time = timer.elapsed_time();
    last_tight_lp   = cur_lp;
    if (tight.usable && tight.lp_obj <= cur_lp * (1. + 1e-3)) {
      cur_primal = std::move(tight.primal);
      cur_dual   = std::move(tight.dual);
    }
  };
  // leave room for a final tight solve of a pending improvement
  constexpr double final_reserve = 100.;
  while (!timer.check_time_limit() && !check_b_b_preemption() &&
         timer.remaining_time() > margin + 2.) {
    if (cur_lp < last_tight_lp * (1. - 1e-4) && timer.remaining_time() < final_reserve) { break; }
    ++n_iter;
    // price the binaries: reduced cost c_j - A_j^T y on the fixed LP's duals
    i_t n_open_cand = 0, n_close_cand = 0;
    std::vector<std::pair<f_t, i_t>> cands;
    for (auto v : bins) {
      f_t aty = 0;
      for (i_t kk = fp_host.int_offsets[v]; kk < fp_host.int_offsets[v + 1]; ++kk) {
        aty += fp_host.int_vals[kk] * cur_dual[fp_host.row_orig[fp_host.int_rows[kk]]];
      }
      red[v]       = h_obj[v] - aty;
      const bool on = cur_assign[v] > 0.5;
      const f_t gain = on ? red[v] : -red[v];
      if (gain > 0) {
        on ? n_close_cand++ : n_open_cand++;
        if (tabu_until[v] <= n_iter) { cands.push_back({gain, v}); }
      }
    }
    if (cands.empty()) {
      CUOPT_LOG_INFO("Dual flip iter %d: no candidates (open=%d close=%d) elapsed=%.2f",
                     n_iter,
                     n_open_cand,
                     n_close_cand,
                     timer.elapsed_time());
      break;
    }
    std::sort(cands.begin(), cands.end(), [](auto& a, auto& b) { return a.first > b.first; });
    const int kb = std::min<int>(k, (int)cands.size());
    std::vector<char> in_batch(n_vars, 0);
    std::vector<i_t> order;
    order.reserve(h_int.size());
    f_t pred_gain = 0;
    for (int b = 0; b < kb; ++b) {
      i_t v       = cands[b].second;
      in_batch[v] = 1;
      order.push_back(v);
      pred_gain += cands[b].first;
    }
    // the rest keeps its values, most strongly supported by the duals first (the least
    // supported ones are the first to be pushed out by propagation)
    std::vector<std::pair<f_t, i_t>> rest;
    rest.reserve(h_int.size());
    for (auto v : h_int) {
      if (in_batch[v]) continue;
      f_t keep = 0;
      if (is_bin[v]) { keep = cur_assign[v] > 0.5 ? -red[v] : red[v]; }
      rest.push_back({keep, v});
    }
    std::stable_sort(rest.begin(), rest.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (auto& p : rest) {
      order.push_back(p.second);
    }
    for (auto v : h_int) {
      pref[v] = in_batch[v] ? 1. - std::round(cur_assign[v]) : std::round(cur_assign[v]);
    }
    i_t n_flipped = 0, n_implied = 0;
    const double t_dive = timer.elapsed_time();
    i_t n_conflicts     = fp_host.dive(order, pref, n_flipped, n_implied);
    i_t n_open = 0, n_close = 0, n_batch_done = 0;
    std::vector<f_t> cand_assign = cur_assign;
    for (auto v : h_int) {
      f_t nv = fp_host.col_lb[v];
      if (std::abs(nv - cur_assign[v]) > 0.5) {
        nv > cur_assign[v] ? n_open++ : n_close++;
        n_batch_done += in_batch[v];
      }
      cand_assign[v] = nv;
    }
    if (n_conflicts > 0 || n_open + n_close == 0) {
      for (int b = 0; b < kb; ++b) {
        tabu_until[cands[b].second] = n_iter + tabu_len;
      }
      CUOPT_LOG_INFO(
        "Dual flip iter %d: k=%d dive rejected conflicts=%d changes=%d dive=%.3fs elapsed=%.2f",
        n_iter,
        kb,
        n_conflicts,
        n_open + n_close,
        timer.elapsed_time() - t_dive,
        timer.elapsed_time());
      k = std::max(1, k / 2);
      continue;
    }
    const f_t tlimit = std::min<f_t>(25., timer.remaining_time() - margin);
    if (tlimit < 2.) { break; }
    auto ev = evaluate(cand_assign, screen_tol, tlimit, false);
    const bool accept = ev.usable && ev.optimal && ev.lp_obj < cur_lp - 1e-6 * std::abs(cur_lp);
    CUOPT_LOG_INFO(
      "Dual flip iter %d: k=%d cands=%zu (open=%d close=%d) pred_gain=%g changes open=%d close=%d "
      "batch_done=%d lp=%g (cur %g, %+.4f%%) optimal=%d iters=%d eval=%.2fs accept=%d "
      "elapsed=%.2f",
      n_iter,
      kb,
      cands.size(),
      n_open_cand,
      n_close_cand,
      pred_gain,
      n_open,
      n_close,
      n_batch_done,
      ev.lp_obj,
      cur_lp,
      100. * (ev.lp_obj - cur_lp) / std::abs(cur_lp),
      (int)ev.optimal,
      ev.iters,
      ev.time,
      (int)accept,
      timer.elapsed_time());
    if (!accept) {
      for (int b = 0; b < kb; ++b) {
        tabu_until[cands[b].second] = n_iter + tabu_len;
      }
      k = std::max(1, k / 2);
      continue;
    }
    n_accept++;
    cur_assign = std::move(cand_assign);
    cur_primal = std::move(ev.primal);
    cur_dual   = std::move(ev.dual);
    cur_lp     = ev.lp_obj;
    k          = std::min(k_max, 2 * k);
    // tight solve + repair + publish once the screened LP gain is worth it
    const bool worth = cur_lp < last_tight_lp * (1. - 0.01) &&
                       timer.elapsed_time() - last_tight_time > 30.;
    if (worth && timer.remaining_time() > margin + 10.) { run_tight(); }
  }
  if (cur_lp < last_tight_lp * (1. - 1e-4) && !timer.check_time_limit() &&
      !check_b_b_preemption() && timer.remaining_time() > margin + 10.) {
    run_tight();
  }
  CUOPT_LOG_INFO("Dual flip end: iters=%d accepted=%d cur_lp=%g best_pub=%g time=%.2fs elapsed=%.2f",
                 n_iter,
                 n_accept,
                 cur_lp,
                 best_pub,
                 timer.elapsed_time() - t_start,
                 timer.elapsed_time());
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

  if (use_lp) {
    // LP-guided pattern search: fix-and-propagate dives under several orders / value
    // preferences, each pattern ranked by a short cost-aware fixed LP (its objective settles
    // within a few thousand PDLP iterations even when the residual stalls); improvements are
    // repaired and published right away; the best pattern then gets the tight polish and the
    // flip loop. On CCBJI, LP-value-descending up-rounding is by far the cheapest family.
    auto h_obj  = cuopt::host_copy(problem_ptr->objective_coefficients, stream);
    auto h_dual = cuopt::host_copy(lp_dual_optimal_solution, stream);
    std::vector<f_t> red(n_vars, 0);
    std::vector<i_t> uplocks(n_vars, 0);
    for (auto v : h_int) {
      f_t aty = 0;
      for (i_t kk = fp_host.int_offsets[v]; kk < fp_host.int_offsets[v + 1]; ++kk) {
        const i_t r = fp_host.int_rows[kk];
        const f_t a = fp_host.int_vals[kk];
        aty += a * h_dual[fp_host.row_orig[r]];
        if ((a > 0 && std::isfinite(fp_host.row_ub[r])) ||
            (a < 0 && std::isfinite(fp_host.row_lb[r]))) {
          uplocks[v]++;
        }
      }
      red[v] = h_obj[v] - aty;
    }
    enum order_kind_t { ORD_LPDESC, ORD_LPDESC_NOISE, ORD_REDCOST, ORD_UPLOCKS };
    enum pref_kind_t { PREF_UP, PREF_UP_NEGRED, PREF_UP_THRESH };
    struct cand_t {
      std::string name;
      int lp_src;  // 0 = root LP point, 1 = continued root LP point
      order_kind_t ord;
      pref_kind_t pref;
      double param;
      unsigned seed;
    };
    std::vector<std::vector<f_t>> seen;
    f_t best_lp     = std::numeric_limits<f_t>::infinity();
    int n_eval      = 0;
    std::string best_name;
    std::vector<f_t> best_repaired;
    std::vector<f_t> best_primal, best_dual;  // phase-A fixed-LP iterate of the best pattern
    constexpr f_t rank_tol    = 1e-4;
    constexpr f_t rank_tlimit = 8.;
    const std::vector<f_t>& vub_ref = h_vub;
    auto run_candidate = [&](const cand_t& c, const std::vector<f_t>& x) {
      if (fpc_timer.check_time_limit() || timer.check_time_limit() || check_b_b_preemption() ||
          timer.remaining_time() < 60.) {
        return;
      }
      std::vector<i_t> ord(h_int.begin(), h_int.end());
      std::vector<f_t> key(n_vars, 0);
      std::mt19937 g(c.seed);
      std::uniform_real_distribution<double> U(0., 1.);
      for (auto v : h_int) {
        switch (c.ord) {
          case ORD_LPDESC: key[v] = x[v]; break;
          case ORD_LPDESC_NOISE: key[v] = x[v] + c.param * U(g); break;
          case ORD_REDCOST: key[v] = -red[v]; break;
          case ORD_UPLOCKS: key[v] = -(f_t)uplocks[v] + 1e-3 * x[v]; break;
        }
      }
      std::stable_sort(ord.begin(), ord.end(), [&](i_t a, i_t b) { return key[a] > key[b]; });
      std::vector<f_t> pref(n_vars, 0.);
      for (auto v : h_int) {
        f_t fl = std::floor(x[v] + int_tol);
        f_t fr = x[v] - fl;
        switch (c.pref) {
          case PREF_UP: pref[v] = fr > int_tol ? fl + 1 : fl; break;
          case PREF_UP_NEGRED:
            pref[v] = (fr > int_tol || red[v] < -c.param) ? std::min(fl + 1, vub_ref[v]) : fl;
            break;
          case PREF_UP_THRESH: pref[v] = fr > c.param ? fl + 1 : fl; break;
        }
      }
      const double t_dive = timer.elapsed_time();
      i_t n_flipped = 0, n_implied = 0;
      fp_host.work    = 0;
      i_t n_conflicts = fp_host.dive(ord, pref, n_flipped, n_implied);
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
        "Pattern dive %s: time=%.3fs conflicts=%d flipped=%d implied_off_pref=%d ones=%d "
        "int_cost=%g dup=%d elapsed=%.2f",
        c.name.c_str(),
        timer.elapsed_time() - t_dive,
        n_conflicts,
        n_flipped,
        n_implied,
        n_ones,
        int_cost,
        (int)dup,
        timer.elapsed_time());
      if (dup || n_conflicts > 0) { return; }
      seen.push_back(pat);
      std::vector<f_t> assign = x;
      for (size_t q = 0; q < h_int.size(); ++q) {
        assign[h_int[q]] = pat[q];
      }
      auto ev = eval_fixed_pattern(
        assign, rank_tol, rank_tlimit, nullptr, nullptr, true, 20., "pattern_rank", best_lp);
      n_eval++;
      const bool better = ev.usable && ev.lp_obj < best_lp;
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
        best_primal   = std::move(ev.primal);
        best_dual     = std::move(ev.dual);
      }
    };
    // phase 1: the cheapest known family first so it publishes early, then its neighbours
    const std::vector<cand_t> phase1 = {
      {"lpdesc_up", 0, ORD_LPDESC, PREF_UP, 0., 0u},
      {"lpdesc_up_negred", 0, ORD_LPDESC, PREF_UP_NEGRED, 1e-9, 0u},
      {"lpdesc_noise05_up", 0, ORD_LPDESC_NOISE, PREF_UP, 0.05, 11u},
      {"lpdesc_noise20_up", 0, ORD_LPDESC_NOISE, PREF_UP, 0.2, 12u},
      {"uplocks_up", 0, ORD_UPLOCKS, PREF_UP, 0., 0u},
      {"lpdesc_up_t05", 0, ORD_LPDESC, PREF_UP_THRESH, 0.05, 0u},
      {"redcost_up", 0, ORD_REDCOST, PREF_UP, 0., 0u}};
    for (auto& c : phase1) {
      run_candidate(c, h_lp);
    }
    // phase 2: continue the (time-limited) root LP warm-started, re-derive the LP-guided
    // patterns from the more converged point
    std::vector<f_t> h_lp2;
    if (!fpc_timer.check_time_limit() && !timer.check_time_limit() && !check_b_b_preemption() &&
        timer.remaining_time() > 300.) {
      const double t_lp2 = timer.elapsed_time();
      auto snapshot      = make_root_lp_snapshot(*problem_ptr);
      std::vector<var_t> h_types(problem_ptr->n_variables, var_t::CONTINUOUS);
      snapshot.set_variable_types(h_types.data(), problem_ptr->n_variables);
      pdlp_solver_settings_t<i_t, f_t> lp_settings{};
      lp_settings.set_optimality_tolerance(1e-4);
      lp_settings.time_limit = 30.;
      lp_settings.method     = method_t::PDLP;
      lp_settings.presolver  = presolver_t::None;
      std::atomic<int> local_halt{0};
      lp_settings.concurrent_halt = &local_halt;
      lp_settings.inside_mip      = false;
      lp_settings.num_gpus        = context.settings.num_gpus;
      rmm::device_uvector<f_t> warm_x(lp_optimal_solution, stream);
      clamp_within_var_bounds(warm_x, problem_ptr, handle_ptr);
      handle_ptr->sync_stream();
      lp_settings.set_initial_primal_solution(warm_x.data(), (i_t)warm_x.size(), stream);
      lp_settings.set_initial_dual_solution(
        lp_dual_optimal_solution.data(), (i_t)lp_dual_optimal_solution.size(), stream);
      auto lp2       = solve_lp(snapshot, lp_settings);
      const auto& li = lp2.get_additional_termination_information();
      const bool ok  = lp2.get_primal_solution().size() == (size_t)n_vars;
      CUOPT_LOG_INFO(
        "Pattern root LP continuation: status=%s obj=%g iters=%d l2_primal_res=%g "
        "l2_dual_res=%g time=%.2fs usable=%d elapsed=%.2f",
        lp2.get_termination_status_string().c_str(),
        (double)lp2.get_objective_value(),
        li.number_of_steps_taken,
        li.l2_primal_residual,
        li.l2_dual_residual,
        timer.elapsed_time() - t_lp2,
        (int)ok,
        timer.elapsed_time());
      if (ok) {
        h_lp2 = cuopt::host_copy(lp2.get_primal_solution(), stream);
        for (i_t j = 0; j < n_vars; ++j) {
          h_lp2[j] = std::min(std::max(h_lp2[j], h_vlb[j]), h_vub[j]);
        }
        i_t n_frac2 = 0, n_changed = 0;
        for (auto v : h_int) {
          f_t xv = h_lp2[v];
          if ((xv - std::floor(xv) > int_tol) && (std::ceil(xv) - xv > int_tol)) n_frac2++;
          n_changed += (h_lp2[v] > int_tol) != (h_lp[v] > int_tol);
        }
        CUOPT_LOG_INFO("Pattern root LP continuation: lp_fractional=%d support_changes=%d",
                       n_frac2,
                       n_changed);
        const std::vector<cand_t> phase2 = {
          {"lp2_lpdesc_up", 1, ORD_LPDESC, PREF_UP, 0., 0u},
          {"lp2_lpdesc_noise05_up", 1, ORD_LPDESC_NOISE, PREF_UP, 0.05, 21u}};
        for (auto& c : phase2) {
          run_candidate(c, h_lp2);
        }
      }
    }
    if (best_repaired.empty()) {
      CUOPT_LOG_INFO("Pattern search: no repaired pattern (evaluated %d) elapsed=%.2f",
                     n_eval,
                     timer.elapsed_time());
      return false;
    }
    CUOPT_LOG_INFO("Pattern search best: %s lp_obj=%g evaluated=%d elapsed=%.2f",
                   best_name.c_str(),
                   best_lp,
                   n_eval,
                   timer.elapsed_time());
    // polish of the best pattern: continue its ranking LP (PID, warm, no presolve) so the
    // objective settles further, then frozen-weight pushes + repair
    solution_t<i_t, f_t> to_polish(*problem_ptr);
    to_polish.copy_new_assignment(best_repaired);
    to_polish.compute_feasibility();
    flip_state = flip_lp_state_t{};
    polish_continuous(to_polish, 120., &flip_state, 0, &best_primal, &best_dual, -1, 30.);
    return true;
  }

  // with the LP: nearest / up-biased / down-biased; without: upper bounds / lower bounds
  const int n_attempts = use_lp ? 3 : 2;
  bool any_found       = false;
  // LP-guided order: up-biased first (its polished fixed-LP optimum was the cheapest on CCBJI)
  const int lp_attempt_order[3] = {1, 0, 2};
  for (int attempt_idx = 0; attempt_idx < n_attempts; ++attempt_idx) {
    const int attempt = use_lp ? lp_attempt_order[attempt_idx] : attempt_idx;
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
        if (use_lp) {
          // every LP-guided rounding gets a cost-aware polish; keep going through the remaining
          // value preferences since the fixed-LP optimum depends on the binary pattern
          // the first LP-guided rounding (up-biased) is polished and handed to the dual-guided
          // flip loop, which gets the remaining time instead of the other value preferences
          flip_state = flip_lp_state_t{};
          polish_continuous(to_polish, 300., &flip_state);
          return true;
        }
      }
      return true;
    }
  }
  if (any_found) { return true; }
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
  // cost-aware polish of the LP-free incumbent before the root LP
  if (!fj_only_run && !simplex_solution_exists.load() && !check_b_b_preemption()) {
    population.add_external_solutions_to_population();
    if (population.is_feasible()) {
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
    if (!timer.check_time_limit() && !check_b_b_preemption()) {
      dual_flip_loop();
      population.add_external_solutions_to_population();
    }
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
