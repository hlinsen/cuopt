/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_cpufj.cuh"

#include "cpu/search/api.hpp"

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/splitmix64.hpp>

#include <omp.h>

#include <algorithm>
#include <string>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::early_cpufj_t(
  const optimization_problem_t<i_t, f_t>& op_problem,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  early_incumbent_callback_t<f_t> incumbent_callback,
  uint64_t seed)
  : early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>(op_problem, std::move(incumbent_callback)),
    problem_ptr_(&op_problem),
    tolerances_(tolerances),
    seed_(seed)
{
}

template <typename i_t, typename f_t>
early_cpufj_t<i_t, f_t>::~early_cpufj_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::start(int n_lanes,
                                    bool low_latency,
                                    const std::vector<f_t>& incumbent_assignment)
{
  const bool threaded = !omp_in_parallel();
  // 1: presolve, 1: early GPU FJ, 1: early CPU FJ
  if (!climbers_.empty() ||
      (!threaded && omp_get_num_threads() < CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT)) {
    return;
  }

  this->preemption_flag_.store(false);
  this->start_time_ = std::chrono::steady_clock::now();

  // Tasks are not preempted, so a lane posted beyond the team size would sit in the queue for the
  // whole of presolve without running an iteration. The lane count is also capped independently of
  // the team size (see CUOPT_MIP_EARLY_CPUFJ_MAX_LANES).
  n_lanes =
    threaded
      ? 1
      : std::clamp(
          n_lanes,
          1,
          std::min(CUOPT_MIP_EARLY_CPUFJ_MAX_LANES,
                   std::max(1, omp_get_num_threads() - CUOPT_MIP_EARLY_CPUFJ_RESERVED_THREADS)));
  cuopt_assert(incumbent_assignment.empty() ||
                 incumbent_assignment.size() == (size_t)this->problem_ptr_->get_n_variables(),
               "incumbent assignment size mismatch");
  cuopt::splitmix64_t seed_rng(seed_);
  lane_base_seed_ = seed_rng.next_i64();
  lane_seeds_.resize(n_lanes);
  for (int k = 0; k < n_lanes; ++k) {
    lane_seeds_[k] = seed_rng.next_i32();
  }
  lane_low_latency_ = low_latency;
  lane_incumbent_   = incumbent_assignment;
  stop_requested_   = false;
  climbers_.resize(n_lanes);
  // Only this download reads the caller's problem, which the caller is free to modify (scale,
  // presolve) once start returns. The host-side build of the lanes runs in the background.
  pending_model_ = std::make_unique<fj_cpu_host_model_t<i_t, f_t>>(
    download_fj_cpu_host_model(*this->problem_ptr_));

  if (!threaded)
    CUOPT_LOG_INFO("Early CPUFJ budget: %d feasibility workers within %d OpenMP threads",
                   n_lanes,
                   omp_get_num_threads());
  CUOPT_LOG_DEBUG("Launching %d early CPUFJ %s", n_lanes, threaded ? "thread" : "tasks");
  if (threaded) {
    worker_ = std::thread([this] { build_and_run_lanes(false); });
    return;
  }
  auto* self  = this;
  auto* token = &builder_token_;
#pragma omp task firstprivate(self) priority(CUOPT_DEFAULT_TASK_PRIORITY) depend(out : *token) \
  default(none)
  {
    self->build_and_run_lanes(true);
#pragma omp taskwait  // the lanes are children of this task
  }
}

// Lane 0 builds the host problem representation and every other lane copies it. A lane starts as
// soon as it is copied; lane 0 starts last, on this thread, so no copy reads a template that is
// already searching.
template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::build_and_run_lanes(bool post_lane_tasks)
{
  const int n_lanes     = (int)climbers_.size();
  auto report_incumbent = [this](f_t solver_obj, const std::vector<f_t>& assignment, double) {
    std::lock_guard<std::mutex> guard(incumbent_mutex_);
    this->try_update_best(solver_obj, assignment);
  };
  auto shared = std::make_shared<fj_cpu_shared_incumbent_t<i_t, f_t>>();

  fj_cpu_climber_t<i_t, f_t>* lane0 = nullptr;
  for (int k = 0; k < n_lanes && !preemption_flag_.load(); ++k) {
    fj_settings_t settings;
    settings.seed = lane_seeds_[k];
    auto climber  = k == 0 ? init_fj_cpu_from_host_model(
                              std::move(*pending_model_), tolerances_, preemption_flag_, settings)
                           : init_fj_cpu_clone(*lane0, preemption_flag_, settings);
    climber->low_latency        = lane_low_latency_;
    const bool incumbent_seeded = !lane_incumbent_.empty() && (n_lanes == 1 || k % 2 == 1);
    if (incumbent_seeded) {
      climber->h_assignment = lane_incumbent_;
      recompute_lhs(*climber);
      climber->h_best_assignment = climber->h_assignment;
    }
    apply_lane_diversification<i_t, f_t>(*climber, k, lane_base_seed_, !incumbent_seeded);
    climber->log_prefix           = "[Early CPUFJ " + std::to_string(k) + "] ";
    climber->improvement_callback = report_incumbent;
    climber->shared_incumbent     = shared;
    auto* lane                    = climber.get();
    {
      std::lock_guard<std::mutex> guard(climbers_mutex_);
      if (stop_requested_) { lane->halted = true; }
      climbers_[k] = std::move(climber);
    }
    if (k == 0) {
      lane0 = lane;
      pending_model_.reset();
    } else if (post_lane_tasks) {
#pragma omp task firstprivate(lane) priority(CUOPT_DEFAULT_TASK_PRIORITY) default(none)
      cpufj_solve(lane);
    }
  }
  if (lane0 != nullptr && !preemption_flag_.load()) { cpufj_solve(lane0); }
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::request_stop()
{
  if (climbers_.empty()) { return; }

  preemption_flag_.store(true);

  // Lanes still being built see the request when they are installed.
  std::lock_guard<std::mutex> guard(climbers_mutex_);
  stop_requested_ = true;
  for (auto& climber : climbers_) {
    if (climber) { climber->halted = true; }
  }
}

template <typename i_t, typename f_t>
void early_cpufj_t<i_t, f_t>::stop()
{
  if (climbers_.empty()) { return; }

  // Every lane is told to stop before the wait.
  request_stop();
  if (worker_.joinable()) {
    worker_.join();
  } else {
    auto* token = &builder_token_;
#pragma omp taskwait depend(in : *token)  // the builder task waits for its lanes
  }
  pending_model_.reset();

  [[maybe_unused]] i_t total_iterations = 0;
  for (const auto& climber : climbers_) {
    if (climber) { total_iterations += climber->iterations; }
  }

  CUOPT_LOG_DEBUG("[Early CPUFJ] Stopped after %d iterations over %d climbers, solution_found=%d",
                  total_iterations,
                  (int)climbers_.size(),
                  this->solution_found_);

  climbers_.clear();
}

template <typename i_t, typename f_t>
std::vector<f_t> early_cpufj_t<i_t, f_t>::to_user_assignment(const std::vector<f_t>& assignment)
{
  return assignment;
}

#if MIP_INSTANTIATE_FLOAT
template class early_cpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_cpufj_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
