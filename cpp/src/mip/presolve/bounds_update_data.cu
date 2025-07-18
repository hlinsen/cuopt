/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "bounds_update_data.cuh"

#include <mip/mip_constants.hpp>
#include <utilities/copy_helpers.hpp>

#include <thrust/uninitialized_fill.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
bounds_update_data_t<i_t, f_t>::bounds_update_data_t(problem_t<i_t, f_t>& problem)
  : bounds_changed(problem.handle_ptr->get_stream()),
    min_activity(problem.n_constraints, problem.handle_ptr->get_stream()),
    max_activity(problem.n_constraints, problem.handle_ptr->get_stream()),
    lb(problem.n_variables, problem.handle_ptr->get_stream()),
    ub(problem.n_variables, problem.handle_ptr->get_stream()),
    changed_constraints(problem.n_constraints, problem.handle_ptr->get_stream()),
    next_changed_constraints(problem.n_constraints, problem.handle_ptr->get_stream()),
    changed_variables(problem.n_variables, problem.handle_ptr->get_stream()),
    implied_lb(problem.n_variables, problem.handle_ptr->get_stream()),
    implied_ub(problem.n_variables, problem.handle_ptr->get_stream())
{
  thrust::uninitialized_fill(problem.handle_ptr->get_thrust_policy(),
                             implied_lb.begin(),
                             implied_lb.end(),
                             -std::numeric_limits<f_t>::infinity());
  thrust::uninitialized_fill(problem.handle_ptr->get_thrust_policy(),
                             implied_ub.begin(),
                             implied_ub.end(),
                             std::numeric_limits<f_t>::infinity());
}

template <typename i_t, typename f_t>
void bounds_update_data_t<i_t, f_t>::resize(problem_t<i_t, f_t>& problem)
{
  min_activity.resize(problem.n_constraints, problem.handle_ptr->get_stream());
  max_activity.resize(problem.n_constraints, problem.handle_ptr->get_stream());
  lb.resize(problem.n_variables, problem.handle_ptr->get_stream());
  ub.resize(problem.n_variables, problem.handle_ptr->get_stream());
  changed_constraints.resize(problem.n_constraints, problem.handle_ptr->get_stream());
  next_changed_constraints.resize(problem.n_constraints, problem.handle_ptr->get_stream());
  changed_variables.resize(problem.n_variables, problem.handle_ptr->get_stream());
  implied_lb.resize(problem.n_variables, problem.handle_ptr->get_stream());
  implied_ub.resize(problem.n_variables, problem.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
typename bounds_update_data_t<i_t, f_t>::view_t bounds_update_data_t<i_t, f_t>::view()
{
  view_t v;
  v.bounds_changed           = bounds_changed.data();
  v.min_activity             = make_span(min_activity);
  v.max_activity             = make_span(max_activity);
  v.lb                       = make_span(lb);
  v.ub                       = make_span(ub);
  v.changed_constraints      = make_span(changed_constraints);
  v.next_changed_constraints = make_span(next_changed_constraints);
  v.changed_variables        = make_span(changed_variables);
  v.implied_lb               = make_span(implied_lb);
  v.implied_ub               = make_span(implied_ub);
  return v;
}

template <typename i_t, typename f_t>
void bounds_update_data_t<i_t, f_t>::init_changed_constraints(const raft::handle_t* handle_ptr)
{
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 1);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_constraints.begin(), changed_constraints.end(), 1);
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
}

template <typename i_t, typename f_t>
void bounds_update_data_t<i_t, f_t>::prepare_for_next_iteration(const raft::handle_t* handle_ptr)
{
  std::swap(changed_constraints, next_changed_constraints);
  handle_ptr->sync_stream();
  thrust::fill(handle_ptr->get_thrust_policy(),
               next_changed_constraints.begin(),
               next_changed_constraints.end(),
               0);
  thrust::fill(
    handle_ptr->get_thrust_policy(), changed_variables.begin(), changed_variables.end(), 0);
}

#if MIP_INSTANTIATE_FLOAT
template class bounds_update_data_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class bounds_update_data_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
