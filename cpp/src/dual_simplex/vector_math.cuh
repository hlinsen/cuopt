/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <cub/cub.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>
namespace cuopt::linear_programming::dual_simplex {

struct norm_inf_max {
  template <typename f_t>
  __device__ __forceinline__ f_t operator()(const f_t& a, const f_t& b) const
  {
    f_t x = std::abs(a);
    f_t y = std::abs(b);
    return x > y ? x : y;
  }
};

template <typename i_t, typename f_t>
f_t vector_norm_inf(const std::vector<f_t>& x, rmm::cuda_stream_view stream_view)
{
  // FIXME: Data should already be on the GPU and tmp storage stored in vector_math class.
  auto d_x   = cuopt::device_copy(x, stream_view);
  auto d_out = rmm::device_scalar<f_t>(stream_view);
  rmm::device_uvector<uint8_t> d_temp_storage(0, stream_view);
  size_t temp_storage_bytes = 0;
  f_t init                  = std::numeric_limits<f_t>::min();
  auto custom_op            = norm_inf_max{};
  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_x.data(),
                            d_out.data(),
                            d_x.size(),
                            custom_op,
                            init);

  d_temp_storage.resize(temp_storage_bytes, stream_view);

  cub::DeviceReduce::Reduce(d_temp_storage.data(),
                            temp_storage_bytes,
                            d_x.data(),
                            d_out.data(),
                            d_x.size(),
                            custom_op,
                            init);
  return d_out.value(stream_view);
}

}  // namespace cuopt::linear_programming::dual_simplex
