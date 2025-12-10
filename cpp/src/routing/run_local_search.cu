/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/run_local_search.hpp>
#include <cuopt/routing/solve.hpp>
#include <routing/solver.hpp>

namespace cuopt {
namespace routing {

template <typename i_t, typename f_t>
assignment_t<i_t> run_local_search(data_model_view_t<i_t, f_t> const& data_model,
                                   solver_settings_t<i_t, f_t> const& settings,
                                   i_t const* solution,
                                   i_t sol_size)
{
  cuopt::routing::solver_t<i_t, f_t> solver(data_model, settings);
  return solver.run_local_search(solution, sol_size);
}

template assignment_t<int> run_local_search(data_model_view_t<int, float> const& data_model,
                                            solver_settings_t<int, float> const& settings,
                                            int const* solution,
                                            int sol_size);

}  // namespace routing
}  // namespace cuopt
