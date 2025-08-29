/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <raft/util/cuda_utils.cuh>

#include <dual_simplex/types.hpp>

namespace cuopt::linear_programming::dual_simplex {


template <typename T>
struct CudaHostAllocator {
  using value_type = T;

  CudaHostAllocator() noexcept {}
  template <class U>
  CudaHostAllocator(const CudaHostAllocator<U>&) noexcept
  {
  }

  T* allocate(std::size_t n)
  {
    T* ptr = nullptr;
    RAFT_CUDA_TRY(cudaMallocHost((void**)&ptr, n * sizeof(T)));
    return ptr;
  }

  void deallocate(T* p, std::size_t) { RAFT_CUDA_TRY(cudaFreeHost(p)); }
};

template <typename T, typename U>
bool operator==(const CudaHostAllocator<T>&, const CudaHostAllocator<U>&) noexcept {
    return true;
}
template <typename T, typename U>
bool operator!=(const CudaHostAllocator<T>&, const CudaHostAllocator<U>&) noexcept {
    return false;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class CudaHostAllocator<double>;
#endif
template class CudaHostAllocator<int>;

}