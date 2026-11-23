/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Stable external include. CPU implementation details and state ownership live below cpu/.
#include <mip_heuristics/feasibility_jump/cpu/state.hpp>

namespace cuopt::mathematical_optimization {
template <typename i_t, typename f_t>
class optimization_problem_t;
}

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
class problem_t;

// CUDA-facing adapters. Their definitions own all interaction with the GPU-backed problem_t;
// cpu/engine.cpp only sees the resulting host climber state.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_standalone(
  problem_t<i_t, f_t>& problem,
  std::atomic<bool>& preemption_flag,
  uint64_t seed,
  fj_settings_t settings = fj_settings_t{});

// Host copy of an optimization_problem_t, taken in one step so the climber can be built from it on
// another thread while the caller goes on modifying the device problem (scaling, presolve).
template <typename i_t, typename f_t>
struct fj_cpu_host_model_t {
  i_t n_variables{0};
  i_t n_constraints{0};
  i_t nnz{0};
  bool maximize{false};
  f_t objective_scaling_factor{1};
  f_t objective_offset{0};
  bool has_zero_coefficients{false};
  std::vector<f_t> coefficients;
  std::vector<i_t> variables;
  std::vector<i_t> offsets;
  std::vector<f_t> objective_coefficients;
  std::vector<f_t> variable_lower_bounds;
  std::vector<f_t> variable_upper_bounds;
  std::vector<f_t> constraint_lower_bounds;
  std::vector<f_t> constraint_upper_bounds;
  std::vector<f_t> constraint_bounds;
  std::vector<char> row_types;
  std::vector<var_t> variable_types;
};

template <typename i_t, typename f_t>
fj_cpu_host_model_t<i_t, f_t> download_fj_cpu_host_model(
  const optimization_problem_t<i_t, f_t>& problem);

// Host-only: safe to run off the thread that owns the problem's stream.
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_host_model(
  fj_cpu_host_model_t<i_t, f_t>&& model,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings = fj_settings_t{});

template <typename i_t, typename f_t>
void build_climber_portfolio(problem_t<i_t, f_t>& problem,
                             std::vector<std::atomic<bool>>& preemption_flags,
                             std::vector<std::unique_ptr<fj_cpu_climber_t<i_t, f_t>>>& climbers,
                             int64_t base_seed,
                             bool low_latency = false);

}  // namespace cuopt::mathematical_optimization::mip
