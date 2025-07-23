/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <routing/adapters/solution_adapter.cuh>
#include <routing/ges/guided_ejection_search.cuh>
#include <routing/ges_solver.cuh>
#include <routing/local_search/local_search.cuh>
#include <routing/routing_test.cuh>
#include <routing/utilities/test_utilities.hpp>

#include <cuopt/routing/solve.hpp>

#include <utilities/copy_helpers.hpp>

#include <gtest/gtest.h>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

constexpr int n_sols = 1;

template <typename i_t, typename f_t, request_t REQUEST>
class cycle_finder_test_t : public routing_test_t<i_t, f_t>,
                            public ::testing::TestWithParam<file_params> {
 public:
  cycle_finder_test_t() {}
  void SetUp() override
  {
    auto param = GetParam();
    auto input = load_routing_file<i_t, f_t>(param.routing_file, 10001);

    this->n_locations        = input.n_locations;
    this->n_vehicles         = input.n_vehicles;
    this->n_orders           = this->n_locations;
    this->x_h                = input.x_h;
    this->y_h                = input.y_h;
    this->demand_h           = input.demand_h;
    this->capacity_h         = input.capacity_h;
    this->earliest_time_h    = input.earliest_time_h;
    this->latest_time_h      = input.latest_time_h;
    this->service_time_h     = input.service_time_h;
    this->pickup_indices_h   = input.pickup_indices_h;
    this->delivery_indices_h = input.delivery_indices_h;
    this->vehicle_types_h.assign(this->n_vehicles, 0);
    this->populate_device_vectors();
    raft::copy(this->pickup_indices_d.data(),
               this->pickup_indices_h.data(),
               this->pickup_indices_h.size(),
               this->stream_view_);
    raft::copy(this->delivery_indices_d.data(),
               this->delivery_indices_h.data(),
               this->delivery_indices_h.size(),
               this->stream_view_);
  }

  bool check_cycle(detail::graph_t<i_t, f_t>& graph, detail::ret_cycles_t<i_t, f_t>& ret)
  {
    auto h_graph  = graph.to_host();
    auto h_cycles = ret.to_host();
    if (!h_cycles.n_cycles) return true;
    f_t cost   = 0;
    auto start = h_cycles.offsets[h_cycles.n_cycles - 1];
    auto end   = h_cycles.offsets[h_cycles.n_cycles];
    std::deque tmp_cycle_path(h_cycles.paths.data() + start, h_cycles.paths.data() + end);
    tmp_cycle_path.push_front(tmp_cycle_path.back());

    for (size_t i = tmp_cycle_path.size() - 1; i > 0; --i) {
      auto node     = tmp_cycle_path[i];
      bool is_cycle = false;
      auto row_id   = h_graph.row_ids[node];
      auto start    = h_graph.rows[row_id];
      auto end      = h_graph.rows[row_id + 1];
      for (int col = start; col < end; ++col) {
        int dst    = h_graph.indices[col];
        f_t weight = h_graph.weights[col];
        if (dst == tmp_cycle_path[i - 1]) {
          cost += weight;
          is_cycle = true;
        }
      }
      if (!is_cycle) return false;
    }
    return cost < -detail::EPSILON;
  }

  void solve(data_model_view_t<i_t, f_t> const& data_model,
             ges_solver_t<i_t, f_t, REQUEST>& ges_solver,
             detail::local_search_t<i_t, f_t, REQUEST>& local_search,
             detail::solution_t<i_t, f_t, REQUEST>& sol)
  {
    ges_solver.pool_allocator.sync_all_streams();
    this->hr_timer_.start("Local search");
    local_search.max_iterations = 1;
    local_search.set_active_weights(detail::default_weights, 0.);
    local_search.run_best_local_search(sol, false, false, true);
    ges_solver.pool_allocator.sync_all_streams();
    this->handle_.sync_stream();
    this->hr_timer_.stop();
    auto routing_solution = ges_solver.get_ges_assignment(sol);
    ASSERT_EQ(routing_solution.get_status(), cuopt::routing::solution_status_t::SUCCESS);
    std::cout << "VN: " << routing_solution.get_vehicle_count()
              << ", Cost: " << routing_solution.get_total_objective() << "\n";

    // FIXME: With new intra operator we don't necessarily execute the cycle finder
    // ASSERT_TRUE(
    //   check_cycle(local_search.move_candidates.graph, local_search.move_candidates.cycles));
    host_assignment_t<i_t> h_routing_solution(routing_solution);
    check_route(data_model, h_routing_solution);
    this->check_time_windows(h_routing_solution, false);
    this->check_capacity(h_routing_solution, this->demand_h, this->capacity_h, this->demand_d);
  }

  void test_pdptw()
  {
    cuopt::routing::data_model_view_t<i_t, f_t> data_model(
      &this->handle_, this->n_locations, this->n_vehicles, this->n_orders);
    data_model.add_cost_matrix(this->cost_matrix_d.data());
    if constexpr (REQUEST == request_t::PDP) {
      data_model.set_pickup_delivery_pairs(this->pickup_indices_d.data(),
                                           this->delivery_indices_d.data());
    }
    data_model.add_capacity_dimension("weight", this->demand_d.data(), this->capacity_d.data());
    data_model.set_order_time_windows(this->earliest_time_d.data(), this->latest_time_d.data());
    data_model.set_order_service_times(this->service_time_d.data());

    auto time_limit = 15.f;

    cuopt::routing::solver_settings_t<i_t, f_t> solver_settings;
    solver_settings.set_time_limit(time_limit);

    this->handle_.sync_stream();
    this->hr_timer_.start("Solve");
    ges_solver_t<i_t, f_t, REQUEST> ges_solver(data_model, solver_settings, n_sols, time_limit);
    auto [resource, index] = ges_solver.pool_allocator.resource_pool->acquire();
    const auto start_time  = std::chrono::steady_clock::now();
    detail::solution_t<i_t, f_t, REQUEST> sol{
      ges_solver.problem, 0, ges_solver.pool_allocator.sol_handles[0].get()};
    resource.ges.set_solution_ptr(&sol);
    resource.ges.start_timer(start_time, time_limit);
    resource.ges.route_minimizer_loop();
    sol.compute_cost();
    this->handle_.sync_stream();
    this->hr_timer_.stop();
    std::cout << "Cost before LS: " << sol.get_total_cost(detail::default_weights) << "\n";
    solve(data_model, ges_solver, resource.ls, sol);
    ges_solver.pool_allocator.resource_pool->release(index);
    this->hr_timer_.display(std::cout);
  }
};

typedef cycle_finder_test_t<int, float, request_t::PDP> float_cycle_finder_test_pdp;
typedef cycle_finder_test_t<int, float, request_t::VRP> float_cycle_finder_test_vrp;
TEST_P(float_cycle_finder_test_pdp, CYCLE_FINDER_FLOAT_PDP) { test_pdptw(); }
INSTANTIATE_TEST_SUITE_P(
  float_test,
  float_cycle_finder_test_pdp,
  ::testing::Values(file_params("/home/hugo/rapids/cuopt/datasets/pdptw/LR2_10_2.pdptw")));
TEST_P(float_cycle_finder_test_vrp, CYCLE_FINDER_FLOAT_VRP) { test_pdptw(); }
INSTANTIATE_TEST_SUITE_P(float_test,
                         float_cycle_finder_test_vrp,
                         ::testing::Values(file_params("cvrptw/R2_10_2.TXT")));

}  // namespace test
}  // namespace routing
}  // namespace cuopt
