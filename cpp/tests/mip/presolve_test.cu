/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mip_utils.cuh"

#include <raft/core/handle.hpp>

#include <limits>
#include <vector>

namespace cuopt::linear_programming::test {

cuopt::mps_parser::mps_data_model_t<int, double> create_example_1_problem()
{
  // Create the mathematical optimization problem from the image:
  // Minimize: -2x₁ - 1x₂ + 1x₃ + 1x₄ - 2x₅ + 1x₆
  // Subject to:
  //   +2x₁ + 3x₂ - 2x₃ ≤ 1
  //   +1x₂ - 2x₃ - 1x₄ - 3x₅ + 1x₆ ≤ -11
  //   -1x₃ + 1x₄ + 2x₅ + 3x₆ ≤ 5
  //   +1x₄ - 2x₅ - 1x₆ ≤ 1.5
  // Variable bounds:
  //   0 ≤ x₁, x₂ ≤ 4
  //   1 ≤ x₃, x₄ ≤ 3.5
  //   x₁, x₂ ∈ ℤ (integers)
  //   x₃, x₄ ∈ ℝ (real numbers)
  //   x₅, x₆ ∈ {0, 1} (binary)

  cuopt::mps_parser::mps_data_model_t<int, double> problem;

  // Set up constraint matrix in CSR format
  // Matrix layout (4 constraints × 6 variables):
  // Row 0: +2x₁ + 3x₂ - 2x₃ + 0x₄ + 0x₅ + 0x₆ ≤ 1
  // Row 1: +0x₁ + 1x₂ - 2x₃ - 1x₄ - 3x₅ + 1x₆ ≤ -11
  // Row 2: +0x₁ + 0x₂ - 1x₃ + 1x₄ + 2x₅ + 3x₆ ≤ 5
  // Row 3: +0x₁ + 0x₂ + 0x₃ + 1x₄ - 2x₅ - 1x₆ ≤ 1.5
  std::vector<int> offsets         = {0, 3, 8, 12, 15};  // CSR row offsets
  std::vector<int> indices         = {0,
                                      1,
                                      2,  // Row 0: cols 0,1,2
                                      1,
                                      2,
                                      3,
                                      4,
                                      5,  // Row 1: cols 1,2,3,4,5
                                      2,
                                      3,
                                      4,
                                      5,  // Row 2: cols 2,3,4,5
                                      3,
                                      4,
                                      5};  // Row 3: cols 3,4,5
  std::vector<double> coefficients = {
    2.0,
    3.0,
    -2.0,  // Row 0 coefficients
    1.0,
    -2.0,
    -1.0,
    -3.0,
    1.0,  // Row 1 coefficients
    -1.0,
    1.0,
    2.0,
    3.0,  // Row 2 coefficients
    1.0,
    -2.0,
    -1.0  // Row 3 coefficients
  };

  problem.set_csr_constraint_matrix(coefficients.data(),
                                    coefficients.size(),
                                    indices.data(),
                                    indices.size(),
                                    offsets.data(),
                                    offsets.size());

  // Set constraint bounds (all are ≤ constraints, so upper bounds only)
  std::vector<double> constraint_lower_bounds = {-std::numeric_limits<double>::infinity(),
                                                 -std::numeric_limits<double>::infinity(),
                                                 -std::numeric_limits<double>::infinity(),
                                                 -std::numeric_limits<double>::infinity()};
  std::vector<double> constraint_upper_bounds = {1.0, -11.0, 5.0, 1.5};

  problem.set_constraint_lower_bounds(constraint_lower_bounds.data(),
                                      constraint_lower_bounds.size());
  problem.set_constraint_upper_bounds(constraint_upper_bounds.data(),
                                      constraint_upper_bounds.size());

  // Set variable bounds
  std::vector<double> var_lower_bounds = {
    0.0, 0.0, 1.0, 1.0, 0.0, 0.0};  // x₁,x₂: [0,4], x₃,x₄: [1,3.5], x₅,x₆: [0,1]
  std::vector<double> var_upper_bounds = {4.0, 4.0, 3.5, 3.5, 1.0, 1.0};

  problem.set_variable_lower_bounds(var_lower_bounds.data(), var_lower_bounds.size());
  problem.set_variable_upper_bounds(var_upper_bounds.data(), var_upper_bounds.size());

  // Set objective coefficients (minimize -2x₁ - 1x₂ + 1x₃ + 1x₄ - 2x₅ + 1x₆)
  std::vector<double> objective_coefficients = {-2.0, -1.0, 1.0, 1.0, -2.0, 1.0};
  problem.set_objective_coefficients(objective_coefficients.data(), objective_coefficients.size());

  // Set variable types
  std::vector<char> variable_types = {'I', 'I', 'C', 'C', 'I', 'I'};  // I=Integer, C=Continuous
  problem.set_variable_types(variable_types);

  // Set to minimize (default)
  problem.set_maximize(false);

  // Optional: Set variable names for better debugging
  problem.set_variable_names({"x1", "x2", "x3", "x4", "x5", "x6"});

  return problem;
}

TEST(presolve, test_dominated_columns)
{
  // Set up RAFT handle and context
  const raft::handle_t handle_{};

  // Create the optimization problem
  auto mps_problem = create_example_1_problem();

  mip_solver_settings_t<int, double> settings;
  // set the time limit depending on we are in assert mode or not
#ifdef ASSERT_MODE
  constexpr double test_time_limit = 60.;
#else
  constexpr double test_time_limit = 30.;
#endif

  settings.time_limit                  = test_time_limit;
  settings.mip_scaling                 = false;
  mip_solution_t<int, double> solution = solve_mip(&handle_, mps_problem, settings);

  // Verify objective value
  double expected_objective = -5.5;
  std::cout << "final objective: " << solution.get_objective_value() << std::endl;
  EXPECT_NEAR(solution.get_objective_value(), expected_objective, 1e-6);

  // Check if solution was found
  EXPECT_EQ(solution.get_termination_status(), mip_termination_status_t::Optimal);

  // Get the solution
  auto solution_values = cuopt::host_copy(solution.get_solution());

  // Verify solution values
  std::vector<double> expected_values = {4.0, 0, 3.5, 1.0, 1.0, 0};
  for (size_t i = 0; i < solution_values.size(); i++) {
    EXPECT_NEAR(solution_values[i], expected_values[i], 1e-6);
  }
}

}  // namespace cuopt::linear_programming::test
