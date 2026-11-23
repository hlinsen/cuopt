/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "early_gpufj.cuh"

#include <mip_heuristics/feasibility_jump/feasibility_jump.cuh>
#include <mip_heuristics/mip_constants.hpp>
#include <mip_heuristics/solver_context.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>

#include <raft/core/device_setter.hpp>
#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/fill.h>
#include <rmm/device_uvector.hpp>

#include <limits>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
early_gpufj_t<i_t, f_t>::early_gpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                                       const mip_solver_settings_t<i_t, f_t>& settings,
                                       early_incumbent_callback_t<f_t> incumbent_callback)
  : early_heuristic_t<i_t, f_t, early_gpufj_t<i_t, f_t>>(op_problem, std::move(incumbent_callback))
{
  RAFT_CUDA_TRY(cudaGetDevice(&device_id_));

  // Copy the original problem onto our own handle, so the derived solver runs on a dedicated
  // stream (prevents graph capture conflicts). Only this copy reads the caller's problem, which
  // the caller is free to modify once the constructor returns; the preprocessing runs in start's
  // task.
  {
    problem_t<i_t, f_t> temp_problem(op_problem, settings.get_tolerances(), false);
    temp_problem.handle_ptr->sync_stream();
    problem_ptr_ = std::make_unique<problem_t<i_t, f_t>>(temp_problem, &handle_);
    handle_.sync_stream();
  }

  context_ptr_ =
    std::make_unique<mip_solver_context_t<i_t, f_t>>(&handle_, problem_ptr_.get(), settings);
}

template <typename i_t, typename f_t>
early_gpufj_t<i_t, f_t>::~early_gpufj_t()
{
  stop();
}

template <typename i_t, typename f_t>
void early_gpufj_t<i_t, f_t>::start()
{
  // 1: presolve, 1: early GPU FJ, 1: early CPU FJ
  if (started_ || omp_get_num_threads() < CUOPT_MIP_EARLY_GPUFJ_REQUIRED_THREAD_COUNT) { return; }
  started_ = true;

  this->start_time_ = std::chrono::steady_clock::now();

  CUOPT_LOG_DEBUG("Launching early GPUFJ task");

  auto* self  = this;
  auto* token = &task_token_;
#pragma omp task default(none) firstprivate(self) priority(CUOPT_DEFAULT_TASK_PRIORITY) \
  depend(out : *token)
  {
    raft::device_setter guard(self->device_id_);
    self->prepare_and_solve();
  }
}

// The preprocessing and the solver setup run in the task too: on large models they take seconds
// that would otherwise delay presolve.
template <typename i_t, typename f_t>
void early_gpufj_t<i_t, f_t>::prepare_and_solve()
{
  if (context_ptr_->preempt_heuristic_solver_.load()) { return; }
  problem_ptr_->preprocess_problem();
  solution_ptr_ = std::make_unique<solution_t<i_t, f_t>>(*problem_ptr_);
  thrust::fill(handle_.get_thrust_policy(),
               solution_ptr_->assignment.begin(),
               solution_ptr_->assignment.end(),
               f_t{0});
  solution_ptr_->clamp_within_bounds();

  fj_settings_t fj_settings;
  fj_settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = std::numeric_limits<int>::max();
  fj_settings.time_limit             = std::numeric_limits<f_t>::infinity();
  fj_settings.iteration_limit        = std::numeric_limits<int>::max();
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;

  fj_ptr_ = std::make_unique<fj_t<i_t, f_t>>(*context_ptr_, fj_settings, rng_id_t::early_gpufj);
  if (context_ptr_->preempt_heuristic_solver_.load()) { return; }
  fj_ptr_->solve(*solution_ptr_, [this](f_t user_obj, const std::vector<f_t>& h_assignment) {
    f_t solver_obj = this->problem_ptr_->get_solver_obj_from_user_obj(user_obj);
    this->try_update_best(solver_obj, h_assignment);
  });
}

template <typename i_t, typename f_t>
void early_gpufj_t<i_t, f_t>::stop()
{
  if (!started_) { return; }

  context_ptr_->preempt_heuristic_solver_.store(true);
  auto* token = &task_token_;
#pragma omp taskwait depend(in : *token)  // Wait for the early GPU FJ task to finish

  CUOPT_LOG_DEBUG("[Early GPU FJ] Stopped, solution_found=%d", this->solution_found_);

  fj_ptr_.reset();
  started_ = false;
}

template <typename i_t, typename f_t>
std::vector<f_t> early_gpufj_t<i_t, f_t>::to_user_assignment(const std::vector<f_t>& assignment)
{
  // Uses a private CUDA stream to avoid racing with the FJ solver's stream.
  RAFT_CUDA_TRY(cudaSetDevice(device_id_));
  auto stream = handle_.get_stream();
  rmm::device_uvector<f_t> d_assignment(assignment.size(), stream);
  raft::copy(d_assignment.data(), assignment.data(), assignment.size(), stream);
  problem_ptr_->post_process_assignment(d_assignment, true, stream);
  return cuopt::host_copy(d_assignment, stream);
}

#if MIP_INSTANTIATE_FLOAT
template class early_gpufj_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class early_gpufj_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
