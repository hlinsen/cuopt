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

// Builds and launches a CUDA graph in which an unconditionally-captured
// prologue is followed by a conditional (IF/WHILE) node whose body is also
// populated by stream capture. Existing kernel-launch code is reused verbatim.
//
// This exists to remove a device->host sync from a local-search step: instead
// of copying a move counter to the host to decide whether to apply moves, a
// kernel in the prologue sets the condition on the device via
// cudaGraphSetConditional(handle, cond), and the IF node decides on the device
// whether to run the (apply-moves) body -- all from a single graph launch.
//
// Three-phase usage, all on the same `stream` (mirrors the stream-capture
// pattern in NVIDIA/cuda-samples graphConditionalNodes):
//
//   conditional_cuda_graph_t g;
//   auto handle = g.begin_prologue_capture(stream, cudaGraphCondTypeIf);
//   //   launch prologue kernels on `stream`; one must call
//   //   cudaGraphSetConditional(handle, cond) (e.g. cond = moves_found > 0)
//   g.begin_body_capture(stream);   // append IF node, open body capture
//   //   launch body kernels on `stream`; they run iff cond != 0
//   g.end_body_capture(stream);     // end capture + instantiate
//   g.launch_graph(stream);
//
// Notes / constraints (see CUDA Programming Guide, "Conditional nodes"):
//   - A graph containing conditional nodes may only have a single live
//     instantiation and cannot be cloned, so we rebuild on every
//     begin_prologue_capture() rather than using cudaGraphExecUpdate.
//   - Body capture must stay capture-safe: no host syncs, and any thrust calls
//     must use the non-synchronizing policy (see get_thrust_policy_nosync) with
//     pre-allocated scratch so nothing allocates mid-capture.
//   - This is not a thread-safe class; be careful on multi-threading.
struct conditional_cuda_graph_t {
  // Phase 1: start capturing the unconditional prologue into a fresh graph and
  // create the conditional handle. Returns the handle; launch the prologue
  // kernels on `stream` afterwards, one of which must call
  // cudaGraphSetConditional(handle, cond).
  cudaGraphConditionalHandle begin_prologue_capture(rmm::cuda_stream_view stream,
                                                    cudaGraphConditionalNodeType type)
  {
    // Tear down any previous build (conditional graphs can't be exec-updated).
    reset();
    conditional_type = type;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    capture_started = true;

    // Obtain the graph being captured so the conditional handle can be created
    // against it before any prologue kernel references it.
    cudaStreamCaptureStatus status;
    cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, nullptr, nullptr, nullptr);
    cudaGraphConditionalHandleCreate(&handle, graph);
    return handle;
  }

  // Phase 2: append the conditional node after the captured prologue (depending
  // on the prologue's frontier), then begin capturing the conditional body on
  // the same stream. Launch body kernels on `stream` afterwards.
  void begin_body_capture(rmm::cuda_stream_view stream)
  {
    cuopt_assert(capture_started, "begin_prologue_capture was not called!");

    // Dependencies = current capture frontier (terminal prologue nodes).
    cudaStreamCaptureStatus status;
    const cudaGraphNode_t* dependencies = nullptr;
    size_t num_dependencies             = 0;
    cudaStreamGetCaptureInfo(
      stream, &status, nullptr, &graph, &dependencies, nullptr, &num_dependencies);

    cudaGraphNodeParams params = {};
    params.type                = cudaGraphNodeTypeConditional;
    params.conditional.handle  = handle;
    params.conditional.type    = conditional_type;
    params.conditional.size    = 1;
    cudaGraphAddNode(&conditional_node, graph, dependencies, nullptr, num_dependencies, &params);

    // The body graph is owned by the conditional node (valid for its lifetime);
    // we never destroy it directly -- destroying `graph` releases it.
    body_graph = params.conditional.phGraph_out[0];

    // Account for the node we added manually so capture stays consistent.
    cudaStreamUpdateCaptureDependencies(
      stream, &conditional_node, nullptr, 1, cudaStreamSetCaptureDependencies);

    // Close the prologue capture, then capture the body into the conditional
    // body graph on the same stream.
    cudaStreamEndCapture(stream, &graph);
    cudaStreamBeginCaptureToGraph(
      stream, body_graph, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal);
  }

  // Phase 3: end body capture and instantiate the executable graph.
  void end_body_capture(rmm::cuda_stream_view stream)
  {
    cuopt_assert(capture_started, "begin_prologue_capture was not called before end_body_capture!");
    cuopt_expects(capture_started, error_type_t::RuntimeError, "A runtime error occurred!");
    cudaStreamEndCapture(stream, nullptr);
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

  bool graph_created                            = false;
  bool capture_started                          = false;
  cudaGraph_t graph                             = nullptr;
  cudaGraphExec_t instance                      = nullptr;
  cudaGraphNode_t conditional_node              = nullptr;
  cudaGraph_t body_graph                        = nullptr;
  cudaGraphConditionalHandle handle             = {};
  cudaGraphConditionalNodeType conditional_type = cudaGraphCondTypeIf;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
