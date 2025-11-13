/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/sparse/detail/cusparse_wrappers.h>

#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct cuoptdnvector_t {
  cusparseDnVecDescr_t vec_descr{nullptr};

  cuoptdnvector_t() = default;

  cuoptdnvector_t(cuoptdnvector_t const& other)            = delete;
  cuoptdnvector_t& operator=(cuoptdnvector_t const& other) = delete;

  cuoptdnvector_t& operator=(cuoptdnvector_t&& other)
  {
    if (vec_descr != nullptr) { RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vec_descr)); }
    vec_descr       = other.vec_descr;
    other.vec_descr = nullptr;
    return *this;
  }

  cuoptdnvector_t(const rmm::device_uvector<f_t>& vec)
  {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(
      &vec_descr, vec.size(), const_cast<f_t*>(vec.data())));
  }

  ~cuoptdnvector_t()
  {
    if (vec_descr != nullptr) { RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vec_descr)); }
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
