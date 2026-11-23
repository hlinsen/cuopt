/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>
#include <mip_heuristics/feasibility_jump/fj_cpu.cuh>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class early_cpufj_t : public early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>> {
 public:
  early_cpufj_t(const optimization_problem_t<i_t, f_t>& op_problem,
                const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                early_incumbent_callback_t<f_t> incumbent_callback,
                uint64_t seed);

  ~early_cpufj_t();

  static constexpr const char* name() { return "CPUFJ"; }

  // Callers sharing the team with other work reserve its capacity before sizing this portfolio.
  void start(int n_lanes,
             bool low_latency                             = false,
             const std::vector<f_t>& incumbent_assignment = {});
  // Asks every lane to exit without waiting for it; stop() then only has to wait.
  void request_stop();
  void stop();

  int lane_count() const { return (int)climbers_.size(); }

 private:
  friend class early_heuristic_t<i_t, f_t, early_cpufj_t<i_t, f_t>>;

  std::vector<f_t> to_user_assignment(const std::vector<f_t>& assignment);
  void build_and_run_lanes(bool post_lane_tasks);

  const optimization_problem_t<i_t, f_t>* problem_ptr_{nullptr};
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  // Sized by start; the lanes are filled in by the builder (under climbers_mutex_) as they are made.
  std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>> climbers_;
  std::mutex climbers_mutex_;
  bool stop_requested_{false};
  // Inputs of the builder, set by start before it is launched.
  std::unique_ptr<fj_cpu_host_model_t<i_t, f_t>> pending_model_;
  std::vector<int> lane_seeds_;
  int64_t lane_base_seed_{0};
  bool lane_low_latency_{false};
  std::vector<f_t> lane_incumbent_;
  // Dependence object of the builder task, which completes once all of its lanes have.
  char builder_token_{0};
  std::thread worker_;
  std::atomic<bool> preemption_flag_{false};
  // Explicit seed for this climber's FJ RNG, resolved once from the solve's base seed (see
  // mip_solver_context_t::base_seed) since this heuristic runs before that context exists.
  uint64_t seed_;
  // try_update_best and the incumbent callback behind it are not thread-safe, and every lane
  // reports into them from its own task.
  std::mutex incumbent_mutex_;
};

}  // namespace cuopt::mathematical_optimization::mip
