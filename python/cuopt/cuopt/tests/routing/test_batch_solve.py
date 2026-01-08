# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudf
import numpy as np

from cuopt import routing

import rmm

rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())


def create_tsp_cost_matrix(n_locations):
    """Creates a simple symmetric cost matrix for TSP."""
    cost_matrix = np.zeros((n_locations, n_locations), dtype=np.float32)
    for i in range(n_locations):
        for j in range(n_locations):
            cost_matrix[i, j] = abs(i - j)
    return cudf.DataFrame(cost_matrix)


def test_batch_solve_varying_sizes():
    """Test batch solving TSPs of varying sizes."""
    tsp_sizes = [
        150,
        150,
        150,
        150,
        150,
        150,
    ]

    # Create data models for each TSP
    data_models = []
    for n_locations in tsp_sizes:
        cost_matrix = create_tsp_cost_matrix(n_locations)
        dm = routing.DataModel(n_locations, 1)
        dm.add_cost_matrix(cost_matrix)
        data_models.append(dm)

    # Configure solver settings
    settings = routing.SolverSettings()
    settings.set_time_limit(0.1)

    # Call batch solve
    import time

    start_time = time.time()
    solutions = routing.BatchSolve(data_models, settings)
    end_time = time.time()
    print(f"Batch solve took {end_time - start_time} seconds")
    # for dm in data_models:
    #     start_time = time.time()
    #     solution = routing.Solve(dm, settings)
    #     end_time = time.time()
    #     print(f"Solve took {end_time - start_time} seconds")

    # Verify results
    assert len(solutions) == len(tsp_sizes)
    for i, solution in enumerate(solutions):
        assert solution.get_status() == 0, (
            f"TSP {i} (size {tsp_sizes[i]}) failed"
        )
        assert solution.get_vehicle_count() == 1, (
            f"TSP {i} (size {tsp_sizes[i]}) used multiple vehicles"
        )


test_batch_solve_varying_sizes()
