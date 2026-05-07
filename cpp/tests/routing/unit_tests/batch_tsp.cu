/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/cython/cython.hpp>
#include <cuopt/routing/solve.hpp>
#include <utilities/copy_helpers.hpp>

#include <cuda_runtime_api.h>

#include <raft/core/handle.hpp>

#include <gtest/gtest.h>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <vector>

namespace cuopt {
namespace routing {
namespace test {

using i_t = int;
using f_t = float;

/**
 * @brief Creates a small symmetric cost matrix for TSP
 * @param n_locations Number of locations
 * @return Cost matrix as a flattened vector
 */
std::vector<f_t> create_small_tsp_cost_matrix(i_t n_locations)
{
  std::vector<f_t> cost_matrix(n_locations * n_locations, 0.0f);

  // Create a simple distance matrix based on coordinates on a line
  for (i_t i = 0; i < n_locations; ++i) {
    for (i_t j = 0; j < n_locations; ++j) {
      cost_matrix[i * n_locations + j] = static_cast<f_t>(std::abs(i - j));
    }
  }
  return cost_matrix;
}

/**
 * @brief Test running TSPs of varying sizes in parallel using call_batch_solve API
 */
TEST(batch_tsp, varying_sizes)
{
  std::size_t free_mem{};
  std::size_t total_mem{};
  ASSERT_EQ(cudaSuccess, cudaMemGetInfo(&free_mem, &total_mem));

  auto const initial_pool_size = ((total_mem / 2) / 256) * 256;
  ASSERT_GT(initial_pool_size, 0);
  ASSERT_LE(initial_pool_size, free_mem);

  auto async_upstream = rmm::mr::cuda_async_memory_resource{};
  auto pool_resource =
    rmm::mr::pool_memory_resource{async_upstream, initial_pool_size, initial_pool_size};
  auto pool_resource_ref = rmm::device_async_resource_ref{pool_resource};
  rmm::mr::set_current_device_resource(pool_resource);
  ASSERT_TRUE(rmm::mr::get_current_device_resource_ref() == pool_resource_ref);

  std::vector<i_t> tsp_sizes = {150, 150, 150, 150, 150, 150};
  const i_t n_problems       = static_cast<i_t>(tsp_sizes.size());

  // Create handles and cost matrices for each problem
  std::vector<std::unique_ptr<raft::handle_t>> handles;
  std::vector<rmm::device_uvector<f_t>> cost_matrices_d;
  std::vector<std::unique_ptr<cuopt::routing::data_model_view_t<i_t, f_t>>> data_models;
  std::vector<cuopt::routing::data_model_view_t<i_t, f_t>*> data_model_ptrs;

  for (i_t i = 0; i < n_problems; ++i) {
    handles.push_back(std::make_unique<raft::handle_t>());
    auto& handle = *handles.back();

    auto cost_matrix_h = create_small_tsp_cost_matrix(tsp_sizes[i]);
    cost_matrices_d.push_back(cuopt::device_copy(cost_matrix_h, handle.get_stream()));

    data_models.push_back(std::make_unique<cuopt::routing::data_model_view_t<i_t, f_t>>(
      &handle, tsp_sizes[i], 1, tsp_sizes[i]));
    data_models.back()->add_cost_matrix(cost_matrices_d.back().data());
    data_model_ptrs.push_back(data_models.back().get());
  }

  // Configure solver settings
  cuopt::routing::solver_settings_t<i_t, f_t> settings;
  settings.set_time_limit(0.1);

  // Call batch solve
  auto solutions = cuopt::cython::call_batch_solve(data_model_ptrs, &settings);

  // Verify all solutions
  ASSERT_EQ(solutions.size(), n_problems);
  for (i_t i = 0; i < n_problems; ++i) {
    EXPECT_EQ(solutions[i]->status_, cuopt::routing::solution_status_t::SUCCESS)
      << "TSP " << i << " (size " << tsp_sizes[i] << ") failed";
    EXPECT_EQ(solutions[i]->vehicle_count_, 1)
      << "TSP " << i << " (size " << tsp_sizes[i] << ") used multiple vehicles";
  }
}

}  // namespace test
}  // namespace routing
}  // namespace cuopt
