/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <mip_heuristics/problem/problem.cuh>
#include <utilities/copy_helpers.hpp>

namespace cuopt::mathematical_optimization::mip {

// Own the LP data so LP presolve and scaling cannot mutate the shared MIP root.
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> make_root_lp_snapshot(const problem_t<i_t, f_t>& problem)
{
  optimization_problem_t<i_t, f_t> snapshot(problem.handle_ptr);
  snapshot.set_csr_constraint_matrix(problem.coefficients.data(),
                                     problem.nnz,
                                     problem.variables.data(),
                                     problem.nnz,
                                     problem.offsets.data(),
                                     problem.n_constraints + 1);
  snapshot.set_constraint_lower_bounds(problem.constraint_lower_bounds.data(),
                                       problem.n_constraints);
  snapshot.set_constraint_upper_bounds(problem.constraint_upper_bounds.data(),
                                       problem.n_constraints);
  snapshot.set_objective_coefficients(problem.objective_coefficients.data(), problem.n_variables);
  snapshot.set_objective_offset(problem.presolve_data.objective_offset);
  snapshot.set_objective_scaling_factor(problem.presolve_data.objective_scaling_factor);
  // MIP preprocessing has already normalized the objective coefficients to minimization.
  snapshot.set_maximize(false);
  auto [lower, upper] =
    cuopt::extract_host_bounds<f_t>(problem.variable_bounds, problem.handle_ptr);
  snapshot.set_variable_lower_bounds(lower.data(), problem.n_variables);
  snapshot.set_variable_upper_bounds(upper.data(), problem.n_variables);
  snapshot.set_problem_name("root_lp_snapshot");
  problem.handle_ptr->sync_stream();
  return snapshot;
}

}  // namespace cuopt::mathematical_optimization::mip
