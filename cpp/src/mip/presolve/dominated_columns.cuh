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
#include <mip/presolve/trivial_presolve_helpers.cuh>
#include <mip/problem/problem.cuh>

#include <algorithm>
#include <bitset>
#include <functional>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <cuda/std/functional>

#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::detail {

enum class domination_order_t { REGULAR, NEGATED_XJ, NEGATED_XK, SIZE };

auto constexpr signature_size = 32;
template <typename i_t, typename f_t>
struct dominated_columns_t {
  dominated_columns_t(problem_t<i_t, f_t>& problem_);

  /**
   * @brief Identify implied free and infinite bounds
   *
   */
  std::vector<i_t> identify_candidate_variables(
    typename problem_t<i_t, f_t>::host_view_t& host_problem,
    bound_presolve_t<i_t, f_t>& bounds_presolve,
    std::vector<f_t> const& lb_bars,
    std::vector<f_t> const& ub_bars);
  void compute_signatures(typename problem_t<i_t, f_t>::host_view_t& host_problem);
  std::map<i_t, std::pair<i_t, i_t>> find_shortest_rows(
    typename problem_t<i_t, f_t>::host_view_t& host_problem, std::vector<i_t> const& candidates);
  bool dominates(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                 i_t xj,
                 i_t xk,
                 domination_order_t order);
  void update_variable_bounds(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                              std::vector<f_t> const& lb_bars,
                              std::vector<f_t> const& ub_bars,
                              std::vector<i_t> const& h_variable_mapping,
                              std::vector<f_t>& h_fixed_var_assignment,
                              i_t xj,
                              i_t xk,
                              domination_order_t order);
  void presolve(bound_presolve_t<i_t, f_t>& bounds_presolve);

  problem_t<i_t, f_t>& problem;
  std::vector<std::bitset<signature_size>> signatures;
  std::vector<f_t> out_lb;

  std::vector<f_t> out_ub;
  rmm::cuda_stream_view stream;
};

}  // namespace cuopt::linear_programming::detail
