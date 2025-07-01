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
    bound_presolve_t<i_t, f_t>& bounds_presolve);
  void compute_signatures(typename problem_t<i_t, f_t>::host_view_t& host_problem);
  std::unordered_map<i_t, std::pair<i_t, i_t>> find_shortest_rows(
    typename problem_t<i_t, f_t>::host_view_t& host_problem, std::vector<i_t> const& candidates);
  bool dominates(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                 i_t xj,
                 i_t xk,
                 domination_order_t order);
  void update_variable_bounds(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                              std::vector<i_t> const& h_variable_mapping,
                              std::vector<f_t>& h_fixed_var_assignment,
                              i_t xj,
                              i_t xk,
                              domination_order_t order);

  /**
   * @brief Check if a variable has been inferred (fixed) during presolve
   *
   * @param var_idx Variable index to check
   * @return true if the variable has been inferred, false otherwise
   */
  bool is_variable_inferred(i_t var_idx) const;

  /**
   * @brief Get the inferred value for a variable
   *
   * @param var_idx Variable index
   * @return The inferred value, or NaN if not inferred
   */
  f_t get_inferred_value(i_t var_idx) const;

  void presolve(bound_presolve_t<i_t, f_t>& bounds_presolve);

  /**
   * @brief Remove dominated columns from CSR matrix and update all related data structures
   *
   * @param host_problem Host view of the problem
   * @param dominated_vars Vector of variable indices to remove
   */
  void remove_dominated_columns_from_csr(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                                         std::vector<i_t>& dominated_vars);

  /**
   * @brief Update constraint bounds to account for removed dominated variables
   *
   * @param host_problem Host view of the problem
   * @param dominated_vars Vector of removed variable indices
   */
  void update_constraint_bounds_for_removed_vars(
    typename problem_t<i_t, f_t>::host_view_t& host_problem,
    const std::vector<i_t>& dominated_vars);

  /**
   * @brief Handle free variables that result from removing dominated variables
   *
   * @param host_problem Host view of the problem
   * @param dominated_vars Vector of dominated variables being removed
   * @param var_map Variable map (1 = keep, 0 = remove)
   */
  void handle_free_variables_after_removal(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                                           const std::vector<i_t>& dominated_vars,
                                           rmm::device_uvector<i_t>& var_map);

  /**
   * @brief Handle free variables using the standardize_bounds approach
   *
   * @param host_problem Host view of the problem
   * @param free_vars Vector of newly free variable indices
   * @param var_map Variable map (1 = keep, 0 = remove)
   */
  void handle_free_variables_standardization(
    typename problem_t<i_t, f_t>::host_view_t& host_problem,
    const std::vector<i_t>& free_vars,
    rmm::device_uvector<i_t>& var_map);

  /**
   * @brief Update CSR matrix to handle free variables
   *
   * @param host_problem Host view of the problem
   * @param free_vars Vector of free variable indices
   */
  void update_csr_for_free_variables(typename problem_t<i_t, f_t>::host_view_t& host_problem,
                                     const std::vector<i_t>& free_vars);

  /**
   * @brief Identify and remove dominated variables from the problem
   *
   * @param host_problem Host view of the problem
   * @param bounds_presolve Bounds presolve data
   * @return Vector of removed variable indices
   */
  std::vector<i_t> identify_and_remove_dominated_variables(
    typename problem_t<i_t, f_t>::host_view_t& host_problem,
    bound_presolve_t<i_t, f_t>& bounds_presolve);

  /**
   * @brief Example usage of dominated columns removal functionality
   *
   * This function demonstrates how to use the dominated columns removal
   * functionality in a typical presolve workflow.
   */
  void example_usage();

  problem_t<i_t, f_t>& problem;
  std::vector<std::bitset<signature_size>> signatures;
  std::vector<f_t> out_lb;

  std::vector<f_t> out_ub;
  rmm::cuda_stream_view stream;
};

}  // namespace cuopt::linear_programming::detail
