/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <utilities/macros.cuh>

#include <rmm/cuda_stream_view.hpp>

#pragma once

namespace cuopt {
namespace routing {
namespace detail {

// This is not a thread-safe class, be careful on multi-threading
struct cuda_graph_t {
  void start_capture(rmm::cuda_stream_view stream)
  {
    // Use ThreadLocal mode to allow multi-threaded batch execution
    // Global mode blocks other streams from performing operations during capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    capture_started = true;
  }

  void end_capture(rmm::cuda_stream_view stream)
  {
    cuopt_assert(capture_started, "start_capture was not called before end_capture!");
    cuopt_expects(capture_started, error_type_t::RuntimeError, "A runtime error occurred!");
    cudaStreamEndCapture(stream, &graph);
    capture_started = false;
    if (graph_created) {
      // If the graph fails to update, errorNode will be set to the
      // node causing the failure and updateResult will be set to a
      // reason code.
      cudaGraphExecUpdate(instance, graph, &errorNode, &updateResult);
    }
    // Instantiate during the first iteration or whenever the update
    // fails for any reason
    if (!graph_created || updateResult != cudaGraphExecUpdateSuccess) {
      // If a previous update failed, destroy the cudaGraphExec_t
      // before re-instantiating it
      if (graph_created) { cudaGraphExecDestroy(instance); }
      // Instantiate graphExec from graph. The error node and
      // error message parameters are unused here.
      cudaGraphInstantiate(&instance, graph);
      graph_created = true;
    }
    cudaGraphDestroy(graph);
  }

  void launch_graph(rmm::cuda_stream_view stream) { cudaGraphLaunch(instance, stream); }

  bool graph_created   = false;
  bool capture_started = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaGraphExecUpdateResult updateResult;
  cudaGraphNode_t errorNode;
};

// Builds and launches a CUDA graph whose single top-level node is a conditional
// (IF or WHILE) node. The body of the conditional is populated by capturing
// kernels onto a stream, so existing kernel-launch code can be reused verbatim.
//
// This exists to remove device->host syncs from iterative local-search loops:
// instead of copying a move counter back to the host every iteration to decide
// whether to keep going, a kernel in the body sets the conditional value on the
// device via cudaGraphSetConditional(handle, keep_going), and the WHILE node
// decides on the device whether to iterate again. The whole loop runs from a
// single host-side graph launch with no intermediate sync.
//
// WHILE usage (the body runs at least once, then repeats while the handle != 0):
//
//   conditional_cuda_graph_t g;
//   auto handle = g.begin_body_capture(stream, cudaGraphCondTypeWhile, /*default*/ 1);
//   // launch the loop-body kernels on `stream` with ordinary <<<>>> syntax;
//   // the last device step must set the condition for the next iteration, e.g.
//   //   set_conditional_kernel<<<1, 1, 0, stream>>>(handle, d_n_moves_found);
//   g.end_body_capture(stream);   // ends capture + instantiates
//   g.launch_graph(stream);       // device loops until the body sets handle = 0
//
// Notes / constraints (see CUDA Programming Guide, "Conditional nodes"):
//   - A graph containing conditional nodes may only have a single live
//     instantiation and cannot be cloned, so we rebuild on every
//     begin_body_capture() rather than using cudaGraphExecUpdate.
//   - Body capture must stay capture-safe: no host syncs, and any thrust calls
//     must use the non-synchronizing policy (see get_thrust_policy_nosync) with
//     pre-allocated scratch so nothing allocates mid-capture.
//   - This is not a thread-safe class; be careful on multi-threading.
struct conditional_cuda_graph_t {
  // Creates a fresh parent graph holding one conditional node and opens stream
  // capture into that node's body graph. Returns the conditional handle, which
  // body kernels use with cudaGraphSetConditional(handle, value).
  cudaGraphConditionalHandle begin_body_capture(rmm::cuda_stream_view stream,
                                                cudaGraphConditionalNodeType type,
                                                unsigned int default_launch_value)
  {
    // Tear down any previous build (conditional graphs can't be exec-updated).
    reset();

    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(
      &handle, graph, default_launch_value, cudaGraphCondAssignDefault);

    cudaGraphNodeParams params = {};
    params.type                = cudaGraphNodeTypeConditional;
    params.conditional.handle  = handle;
    params.conditional.type    = type;
    params.conditional.size    = 1;
    cudaGraphAddNode(&conditional_node, graph, nullptr, nullptr, 0, &params);

    // The body graph is owned by the conditional node (valid for its lifetime);
    // we never destroy it directly -- destroying `graph` releases it.
    body_graph = params.conditional.phGraph_out[0];

    // Capture subsequent kernels launched on `stream` into the body graph.
    cudaStreamBeginCaptureToGraph(
      stream, body_graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
    capture_started = true;
    return handle;
  }

  // Ends body capture and instantiates the executable graph.
  void end_body_capture(rmm::cuda_stream_view stream)
  {
    cuopt_assert(capture_started, "begin_body_capture was not called before end_body_capture!");
    cuopt_expects(capture_started, error_type_t::RuntimeError, "A runtime error occurred!");
    // Capture targeted body_graph; the returned graph is the body, which the
    // conditional node already references, so we discard the handle here.
    cudaGraph_t captured = nullptr;
    cudaStreamEndCapture(stream, &captured);
    capture_started = false;

    cudaGraphInstantiate(&instance, graph);
    graph_created = true;
  }

  void launch_graph(rmm::cuda_stream_view stream) { cudaGraphLaunch(instance, stream); }

  // Releases the executable graph and the parent graph (which owns the body).
  void reset()
  {
    if (graph_created) {
      cudaGraphExecDestroy(instance);
      graph_created = false;
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
      graph = nullptr;
    }
    conditional_node = nullptr;
    body_graph       = nullptr;
  }

  ~conditional_cuda_graph_t() { reset(); }

  bool graph_created   = false;
  bool capture_started = false;
  cudaGraph_t graph    = nullptr;
  cudaGraphExec_t instance;
  cudaGraphNode_t conditional_node = nullptr;
  cudaGraph_t body_graph           = nullptr;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
