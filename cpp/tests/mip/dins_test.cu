/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/dins.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::mathematical_optimization::mip::test {

using simplex::variable_type_t;

TEST(DinsTest, BuildsReboundedHardAndSoftNeighborhood)
{
  std::vector<variable_type_t> var_types{variable_type_t::INTEGER,
                                         variable_type_t::INTEGER,
                                         variable_type_t::INTEGER,
                                         variable_type_t::INTEGER,
                                         variable_type_t::INTEGER,
                                         variable_type_t::INTEGER,
                                         variable_type_t::CONTINUOUS};
  std::vector<double> incumbent{5.0, 4.0, 1.0, 0.0, 1.0, 0.0, 2.0};
  std::vector<double> node{4.3, 4.2, 1.0, 0.0, 0.7, 0.6, 1.5};
  std::vector<double> root{4.5, 4.1, 1.0, 0.0, 1.0, 0.0, 1.0};
  std::vector<bool> changed{false, false, false, true, false, false, false};
  std::vector<double> lower{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0};
  std::vector<double> upper{10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0};
  std::vector<bool> bounds_changed(var_types.size(), false);

  auto neighborhood = build_dins_neighborhood<int, double>(
    node, root, incumbent, changed, var_types, 1e-8, 1e-5, 5, lower, upper, bounds_changed);

  EXPECT_DOUBLE_EQ(lower[0], 4.0);
  EXPECT_DOUBLE_EQ(upper[0], 5.0);
  EXPECT_DOUBLE_EQ(lower[1], 4.0);
  EXPECT_DOUBLE_EQ(upper[1], 4.0);
  EXPECT_DOUBLE_EQ(lower[2], 1.0);
  EXPECT_DOUBLE_EQ(upper[2], 1.0);
  EXPECT_DOUBLE_EQ(lower[6], -10.0);
  EXPECT_DOUBLE_EQ(upper[6], 10.0);

  EXPECT_EQ(neighborhood.num_rebounded, 1);
  EXPECT_EQ(neighborhood.num_hard_fixed, 2);
  EXPECT_EQ(neighborhood.soft_variables, (std::vector<int>{3, 4}));
  EXPECT_EQ(neighborhood.soft_coefficients, (std::vector<double>{1.0, -1.0}));
  EXPECT_DOUBLE_EQ(neighborhood.soft_rhs, 4.0);

  raft::handle_t handle;
  simplex::user_problem_t<int, double> problem(&handle);
  problem.num_rows    = 1;
  problem.num_cols    = var_types.size();
  problem.A           = csc_matrix_t<int, double>(1, problem.num_cols, 1);
  problem.A.col_start = {0, 1, 1, 1, 1, 1, 1, 1};
  problem.A.i         = {0};
  problem.A.x         = {1.0};
  problem.rhs         = {0.0};
  problem.row_sense   = {'E'};

  append_local_branching_constraint(problem, neighborhood);
  csr_matrix_t<int, double> rows(problem.num_rows, problem.num_cols, 1);
  ASSERT_EQ(problem.A.to_compressed_row(rows), 0);
  EXPECT_EQ(problem.num_rows, 2);
  EXPECT_EQ(rows.row_length(1), 2);
}

TEST(DinsTest, AdvancesRadiusSearchAccordingToPaper)
{
  dins_search_state_t<int> search(5);

  EXPECT_TRUE(search.advance(false, true, true));
  EXPECT_EQ(search.current_radius, 0);
  EXPECT_TRUE(search.advance(true, true, true));
  EXPECT_EQ(search.current_radius, 5);
  EXPECT_FALSE(search.advance(false, false, true));

  dins_search_state_t<int> empty_soft_set(5);
  EXPECT_FALSE(empty_soft_set.advance(false, true, false));
}

TEST(DinsTest, SchedulesFirstIncumbentAndEveryHundredNodes)
{
  dins_schedule_t<int> schedule;

  EXPECT_FALSE(schedule.should_launch(false, 25));
  EXPECT_TRUE(schedule.should_launch(true, 25));
  schedule.record_launch(25);
  EXPECT_FALSE(schedule.should_launch(true, 124));
  EXPECT_TRUE(schedule.should_launch(true, 125));
}

TEST(DinsTest, ReboundsTowardLargerNodeValue)
{
  std::vector<variable_type_t> var_types{variable_type_t::INTEGER};
  std::vector<double> incumbent{2.0};
  std::vector<double> node{3.2};
  std::vector<double> root{2.5};
  std::vector<bool> changed{false};
  std::vector<double> lower{0.0};
  std::vector<double> upper{10.0};
  std::vector<bool> bounds_changed(1, false);

  auto neighborhood = build_dins_neighborhood<int, double>(
    node, root, incumbent, changed, var_types, 1e-8, 1e-5, 5, lower, upper, bounds_changed);

  EXPECT_DOUBLE_EQ(lower[0], 2.0);
  EXPECT_DOUBLE_EQ(upper[0], 4.0);
  EXPECT_EQ(neighborhood.num_rebounded, 1);
  EXPECT_TRUE(bounds_changed[0]);
}

TEST(DinsTest, AppendsLocalBranchingInequality)
{
  raft::handle_t handle;
  simplex::user_problem_t<int, double> problem(&handle);
  problem.num_rows    = 1;
  problem.num_cols    = 3;
  problem.A           = csc_matrix_t<int, double>(1, 3, 1);
  problem.A.col_start = {0, 1, 1, 1};
  problem.A.i         = {0};
  problem.A.x         = {2.0};
  problem.rhs         = {7.0};
  problem.row_sense   = {'E'};

  dins_neighborhood_t<int, double> neighborhood;
  neighborhood.soft_variables    = {0, 2};
  neighborhood.soft_coefficients = {1.0, -1.0};
  neighborhood.soft_rhs          = 3.0;

  append_local_branching_constraint(problem, neighborhood);

  EXPECT_EQ(problem.num_rows, 2);
  EXPECT_EQ(problem.rhs, (std::vector<double>{7.0, 3.0}));
  EXPECT_EQ(problem.row_sense, (std::vector<char>{'E', 'L'}));

  csr_matrix_t<int, double> rows(problem.num_rows, problem.num_cols, 1);
  ASSERT_EQ(problem.A.to_compressed_row(rows), 0);
  ASSERT_EQ(rows.row_length(1), 2);
  EXPECT_EQ(rows.j[rows.row_start[1]], 0);
  EXPECT_DOUBLE_EQ(rows.x[rows.row_start[1]], 1.0);
  EXPECT_EQ(rows.j[rows.row_start[1] + 1], 2);
  EXPECT_DOUBLE_EQ(rows.x[rows.row_start[1] + 1], -1.0);
}

}  // namespace cuopt::mathematical_optimization::mip::test
