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

#include <mip/presolve/bounds_presolve.cuh>
#include <mip/problem/problem.cuh>

#include <bitset>
#include <unordered_map>
#include <vector>

namespace cuopt::linear_programming::detail {

enum class domination_order_t {
  REGULAR,
  NEGATED_XJ,
  NEGATED_XK,
};

auto constexpr signature_size = 32;
template <typename i_t, typename f_t>
struct dominated_columns_t {
  dominated_columns_t(problem_t<i_t, f_t>& problem_);

  /**
   * @brief Identify implied free and infinite bounds
   *
   */
  void identify_candidate_variables(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                                    bound_presolve_t<i_t, f_t>& bounds_presolve);
  void compute_signatures(typename problem_t<i_t, f_t>::host_view_t& host_problem);
  std::unordered_map<i_t, std::pair<i_t, i_t>> find_shortest_rows(
    typename problem_t<i_t, f_t>::host_view_t& host_problem);
  bool dominates(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                 i_t xj,
                 i_t xk,
                 i_t row,
                 domination_order_t order);
  void update_variable_bounds(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                              i_t xj,
                              i_t xk,
                              domination_order_t order);
  void presolve(bound_presolve_t<i_t, f_t>& bounds_presolve);

  problem_t<i_t, f_t>& problem;
  std::vector<std::bitset<signature_size>> signatures;
  std::vector<f_t> out_lb;

  std::vector<f_t> out_ub;
  std::vector<i_t> candidates;
  rmm::cuda_stream_view stream;
};

}  // namespace cuopt::linear_programming::detail
