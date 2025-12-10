# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# CONFIDENTIAL, provided under NDA.

import cudf

from cuopt import routing


def test_run_local_search():
    """ """

    costs = cudf.DataFrame(
        {
            0: [0, 1, 1, 1, 5, 1, 1, 1, 1, 1],
            1: [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            2: [1, 1, 0, 4, 1, 1, 1, 4, 1, 1],
            3: [2, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            4: [1, 3, 1, 1, 0, 1, 1, 1, 1, 1],
            5: [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            6: [1, 10, 1, 1, 1, 1, 0, 1, 1, 1],
            7: [2, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            8: [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            9: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )

    vehicle_num = 5
    demand = cudf.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    capacities = cudf.Series([2] * vehicle_num)
    sol = cudf.Series([0, 1, 2, 0, 5, 4, 0, 9, 7, 0, 3, 8, 0, 6])
    print(sol)
    d = routing.DataModel(costs.shape[0], vehicle_num)
    d.add_cost_matrix(costs)
    d.add_capacity_dimension("demand", demand, capacities)

    s = routing.SolverSettings()
    s.set_time_limit(10)

    routing_solution = routing.RunLocalSearch(d, s, sol, sol.shape[0])
    routing_solution.display_routes()
    cu_status = routing_solution.get_status()

    assert cu_status == 0


test_run_local_search()
