/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstddef>
#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>

namespace cuopt {
namespace routing {

/**
 * @brief Routing local search function
 *
 * @tparam i_t
 * @tparam f_t
 * @param[in] data_model  input data model of type data_model_view_type
 * @param[in] settings    solver settings of type solver_settings_t
 * @param[in] solution    initial solution to run local search on
 * @param[in] sol_size    size of the solution
 * @return assignment_t<i_t> owning container for the solver output
 */
template <typename i_t, typename f_t>
assignment_t<i_t> run_local_search(
  data_model_view_t<i_t, f_t> const& data_model,
  solver_settings_t<i_t, f_t> const& settings = solver_settings_t<i_t, f_t>{},
  i_t const* solution                         = nullptr,
  i_t sol_size                                = -1);

}  // namespace routing
}  // namespace cuopt
