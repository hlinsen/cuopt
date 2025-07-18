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

#pragma once

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <mip/problem/problem.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct bounds_update_data_t {
  rmm::device_scalar<i_t> bounds_changed;
  rmm::device_uvector<f_t> min_activity;
  rmm::device_uvector<f_t> max_activity;
  rmm::device_uvector<f_t> lb;
  rmm::device_uvector<f_t> ub;
  rmm::device_uvector<i_t> changed_constraints;
  rmm::device_uvector<i_t> next_changed_constraints;
  rmm::device_uvector<i_t> changed_variables;
  rmm::device_uvector<f_t> implied_lb;
  rmm::device_uvector<f_t> implied_ub;

  struct view_t {
    i_t* bounds_changed;
    raft::device_span<f_t> min_activity;
    raft::device_span<f_t> max_activity;
    raft::device_span<f_t> lb;
    raft::device_span<f_t> ub;
    raft::device_span<i_t> changed_constraints;
    raft::device_span<i_t> next_changed_constraints;
    raft::device_span<i_t> changed_variables;
    raft::device_span<f_t> implied_lb;
    raft::device_span<f_t> implied_ub;
  };

  bounds_update_data_t(problem_t<i_t, f_t>& pb);
  void resize(problem_t<i_t, f_t>& problem);
  void init_changed_constraints(const raft::handle_t* handle_ptr);
  void prepare_for_next_iteration(const raft::handle_t* handle_ptr);
  view_t view();
};

}  // namespace cuopt::linear_programming::detail
