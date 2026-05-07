/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/cython/cython.hpp>
#include <cuopt/routing/solve.hpp>
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <rmm/device_buffer.hpp>
#include <routing/generator/generator.hpp>
#include <utilities/driver_helpers.cuh>
#include <utilities/macros.cuh>

#include <omp.h>
#include <chrono>

namespace cuopt {
namespace cython {

template <typename i_t, typename f_t>
void populate_dataset_params(routing::generator::dataset_params_t<i_t, f_t>& params,
                             i_t n_locations,
                             bool asymmetric,
                             i_t dim,
                             routing::demand_i_t const* min_demand,
                             routing::demand_i_t const* max_demand,
                             routing::cap_i_t const* min_capacities,
                             routing::cap_i_t const* max_capacities,
                             i_t min_service_time,
                             i_t max_service_time,
                             f_t tw_tightness,
                             f_t drop_return_trips,
                             i_t n_shifts,
                             i_t n_vehicle_types,
                             i_t n_matrix_types,
                             routing::generator::dataset_distribution_t distrib,
                             f_t center_box_min,
                             f_t center_box_max,
                             i_t seed)
{
  params.n_locations       = n_locations;
  params.asymmetric        = asymmetric;
  params.dim               = dim;
  params.min_demand        = min_demand;
  params.max_demand        = max_demand;
  params.min_capacities    = min_capacities;
  params.max_capacities    = max_capacities;
  params.min_service_time  = min_service_time;
  params.max_service_time  = max_service_time;
  params.tw_tightness      = tw_tightness;
  params.drop_return_trips = drop_return_trips;
  params.n_shifts          = n_shifts;
  params.n_vehicle_types   = n_vehicle_types;
  params.n_matrix_types    = n_matrix_types;
  params.distrib           = distrib;
  params.center_box_min    = center_box_min;
  params.center_box_max    = center_box_max;
  params.seed              = seed;
}

/**
 * @brief Wrapper for vehicle_routing to expose the API to cython
 *
 * @param data_model Composable data model object
 * @param settings  Composable solver settings object
 * @return std::unique_ptr<vehicle_routing_ret_t>
 */
std::unique_ptr<vehicle_routing_ret_t> call_solve(
  routing::data_model_view_t<int, float>* data_model,
  routing::solver_settings_t<int, float>* settings)

{
  auto routing_solution = cuopt::routing::solve(*data_model, *settings);
  vehicle_routing_ret_t vr_ret{
    routing_solution.get_vehicle_count(),
    routing_solution.get_total_objective(),
    routing_solution.get_objectives(),
    std::make_unique<rmm::device_buffer>(routing_solution.get_route().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_order_locations().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_arrival_stamp().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_truck_id().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_node_types().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_unserviced_nodes().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_accepted().release()),
    routing_solution.get_status(),
    routing_solution.get_status_string(),
    routing_solution.get_error_status().get_error_type(),
    routing_solution.get_error_status().what()};
  return std::make_unique<vehicle_routing_ret_t>(std::move(vr_ret));
}

/**
 * @brief Wrapper for batch vehicle_routing to expose the API to cython
 *
 * @param data_models Vector of data model pointers
 * @param settings  Composable solver settings object
 * @return std::vector<std::unique_ptr<vehicle_routing_ret_t>>
 */
std::vector<std::unique_ptr<vehicle_routing_ret_t>> call_batch_solve(
  std::vector<routing::data_model_view_t<int, float>*> data_models,
  routing::solver_settings_t<int, float>* settings)
{
  const std::size_t size = data_models.size();
  std::vector<std::unique_ptr<vehicle_routing_ret_t>> list(size);

  // Use OpenMP for parallel execution
  const int max_thread = std::min(static_cast<int>(size), omp_get_max_threads());

#if CUDART_VERSION >= 13000
  // Set up green contexts for GPU SM partitioning
  std::vector<CUgreenCtx> green_contexts(size);
  std::vector<CUstream> green_streams(size);
  void* cuGetErrorString_func = nullptr;

  cuGetErrorString_func = cuopt::detail::get_driver_entry_point("cuGetErrorString");

  // Get the GPU device resources
  CUdevResource initial_device_GPU_resources = {};
  auto cuDeviceGetDevResource_func =
    cuopt::detail::get_driver_entry_point("cuDeviceGetDevResource");
  int device_id = data_models[0]->get_handle_ptr()->get_device();
  CU_CHECK(reinterpret_cast<decltype(::cuDeviceGetDevResource)*>(cuDeviceGetDevResource_func)(
             device_id, &initial_device_GPU_resources, CU_DEV_RESOURCE_TYPE_SM),
           reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));

  auto total_SMs = initial_device_GPU_resources.sm.smCount;

  // printf("Total SMs: %u\n", total_SMs);
  // Divide SMs equally based on number of orders (data_models)
  auto sms_per_context = std::max(1u, total_SMs / static_cast<unsigned>(size));
  // printf("SMS per context: %u\n", sms_per_context);

  auto cuDevSmResourceSplitByCount_func =
    cuopt::detail::get_driver_entry_point("cuDevSmResourceSplitByCount");
  auto cuDevResourceGenerateDesc_func =
    cuopt::detail::get_driver_entry_point("cuDevResourceGenerateDesc");
  auto cuGreenCtxCreate_func = cuopt::detail::get_driver_entry_point("cuGreenCtxCreate");
  auto cuGreenCtxStreamCreate_func =
    cuopt::detail::get_driver_entry_point("cuGreenCtxStreamCreate");

  // Split resources into n_groups = size (number of problems)
  std::vector<CUdevResource> resources(size);
  auto requested_groups = static_cast<unsigned>(size);
  auto n_groups         = requested_groups;
  auto use_flags        = CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING;
  CUresult split_result = CUDA_SUCCESS;
  do {
    n_groups     = requested_groups;
    split_result = reinterpret_cast<decltype(::cuDevSmResourceSplitByCount)*>(
      cuDevSmResourceSplitByCount_func)(resources.data(),
                                        &n_groups,
                                        &initial_device_GPU_resources,
                                        nullptr,
                                        use_flags,
                                        sms_per_context);
    CU_CHECK(split_result, reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
    if (split_result == CUDA_SUCCESS && n_groups == requested_groups) { break; }
    --sms_per_context;
  } while (sms_per_context > 0);
  RAFT_EXPECTS(split_result == CUDA_SUCCESS && n_groups == requested_groups,
               "Unable to split %u SMs into %u green context groups",
               total_SMs,
               requested_groups);

  // printf(
  //   "Resources were split into %u groups (had requested %zu) with %u SMs each (had requested %u)\n",
  //   n_groups,
  //   size,
  //   resources[0].sm.smCount,
  //   sms_per_context);

  // Create green contexts and streams for each solve
  for (std::size_t i = 0; i < size; ++i) {
    // printf("Problem %zu: %u SMs\n", i, resources[i].sm.smCount);

    CUdevResourceDesc resource_desc;
    CU_CHECK(reinterpret_cast<decltype(::cuDevResourceGenerateDesc)*>(
               cuDevResourceGenerateDesc_func)(&resource_desc, &resources[i], 1),
             reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));

    CU_CHECK(reinterpret_cast<decltype(::cuGreenCtxCreate)*>(cuGreenCtxCreate_func)(
               &green_contexts[i], resource_desc, device_id, CU_GREEN_CTX_DEFAULT_STREAM),
             reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));

    int stream_priority = 0;
    cudaStreamGetPriority(data_models[i]->get_handle_ptr()->get_stream(), &stream_priority);

    CU_CHECK(reinterpret_cast<decltype(::cuGreenCtxStreamCreate)*>(cuGreenCtxStreamCreate_func)(
               &green_streams[i], green_contexts[i], CU_STREAM_NON_BLOCKING, stream_priority),
             reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_func));
  }
#endif

  // int device_id = raft::resource::get_device_id(*(data_models[0]->get_handle_ptr()));

#pragma omp parallel for num_threads(max_thread)
  for (std::size_t i = 0; i < size; ++i) {
    // Required in multi-GPU environments to set the device for each thread
    RAFT_CUDA_TRY(cudaSetDevice(device_id));

    auto old_stream = data_models[i]->get_handle_ptr()->get_stream();
    // Make sure previous operations are finished
    data_models[i]->get_handle_ptr()->sync_stream();

#if CUDART_VERSION >= 13000
    // Set the green context stream for current data model
    rmm::cuda_stream_view green_stream_view(green_streams[i]);
    raft::resource::set_cuda_stream(*(data_models[i]->get_handle_ptr()), green_stream_view);
#endif

    auto routing_solution = cuopt::routing::solve(*data_models[i], *settings);

#if CUDART_VERSION >= 13000
    // Make sure current solve is finished
    cudaStreamSynchronize(green_streams[i]);
#endif

    // Create buffers and reassociate them with the original stream so they
    // outlive the local stream which will be destroyed at end of loop iteration
    auto make_buffer = [old_stream = old_stream](rmm::device_buffer&& buf) {
      buf.set_stream(old_stream);
      return std::make_unique<rmm::device_buffer>(std::move(buf));
    };

    vehicle_routing_ret_t vr_ret{routing_solution.get_vehicle_count(),
                                 routing_solution.get_total_objective(),
                                 routing_solution.get_objectives(),
                                 make_buffer(routing_solution.get_route().release()),
                                 make_buffer(routing_solution.get_order_locations().release()),
                                 make_buffer(routing_solution.get_arrival_stamp().release()),
                                 make_buffer(routing_solution.get_truck_id().release()),
                                 make_buffer(routing_solution.get_node_types().release()),
                                 make_buffer(routing_solution.get_unserviced_nodes().release()),
                                 make_buffer(routing_solution.get_accepted().release()),
                                 routing_solution.get_status(),
                                 routing_solution.get_status_string(),
                                 routing_solution.get_error_status().get_error_type(),
                                 routing_solution.get_error_status().what()};
    list[i] = std::make_unique<vehicle_routing_ret_t>(std::move(vr_ret));

    // Restore the old stream
    raft::resource::set_cuda_stream(*(data_models[i]->get_handle_ptr()), old_stream);
    old_stream.synchronize();
  }

#if CUDART_VERSION >= 13000
  // Clean up green contexts and streams
  {
    auto cuStreamDestroy_func     = cuopt::detail::get_driver_entry_point("cuStreamDestroy");
    auto cuGreenCtxDestroy_func   = cuopt::detail::get_driver_entry_point("cuGreenCtxDestroy");
    auto cuGetErrorString_cleanup = cuopt::detail::get_driver_entry_point("cuGetErrorString");
    for (std::size_t i = 0; i < size; ++i) {
      CU_CHECK(
        reinterpret_cast<decltype(::cuStreamDestroy)*>(cuStreamDestroy_func)(green_streams[i]),
        reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_cleanup));
      CU_CHECK(
        reinterpret_cast<decltype(::cuGreenCtxDestroy)*>(cuGreenCtxDestroy_func)(green_contexts[i]),
        reinterpret_cast<decltype(::cuGetErrorString)*>(cuGetErrorString_cleanup));
    }
  }
#endif

  return list;
}

/**
 * @brief Wrapper for dataset_t to expose the API to cython.
 * @param solver Composable solver object
 */
std::unique_ptr<dataset_ret_t> call_generate_dataset(
  raft::handle_t const& handle, routing::generator::dataset_params_t<int, float> const& params)
{
  auto data           = routing::generator::generate_dataset<int, float>(handle, params);
  auto [x_pos, y_pos] = data.get_coordinates();
  auto& fleet_info    = data.get_fleet_info();
  auto& order_info    = data.get_order_info();

  dataset_ret_t gen_ret{
    std::make_unique<rmm::device_buffer>(x_pos.release()),
    std::make_unique<rmm::device_buffer>(y_pos.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.matrices_.buffer.release()),
    std::make_unique<rmm::device_buffer>(order_info.v_earliest_time_.release()),
    std::make_unique<rmm::device_buffer>(order_info.v_latest_time_.release()),
    std::make_unique<rmm::device_buffer>(
      fleet_info.fleet_order_constraints_.order_service_times.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_earliest_time_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_latest_time_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_drop_return_trip_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_skip_first_trip_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_types_.release()),
    std::make_unique<rmm::device_buffer>(order_info.v_demand_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_capacities_.release())};
  return std::make_unique<dataset_ret_t>(std::move(gen_ret));
}

template void populate_dataset_params<int, float>(
  routing::generator::dataset_params_t<int, float>& params,
  int n_locations,
  bool asymmetric,
  int dim,
  routing::demand_i_t const* min_demand,
  routing::demand_i_t const* max_demand,
  routing::cap_i_t const* min_capacities,
  routing::cap_i_t const* max_capacities,
  int min_service_time,
  int max_service_time,
  float tw_tightness,
  float drop_return_trips,
  int n_shifts,
  int n_vehicle_types,
  int n_matrix_types,
  routing::generator::dataset_distribution_t distrib,
  float center_box_min,
  float center_box_max,
  int seed);

}  // namespace cython
}  // namespace cuopt
