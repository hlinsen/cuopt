/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/sparse/detail/cusparse_wrappers.h>

#include <rmm/device_uvector.hpp>

#define CUOPT_CUSPARSE_TRY_NO_THROW(call)                                    \
  do {                                                                       \
    cusparseStatus_t const status = (call);                                  \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                 \
      std::string msg{};                                                     \
      SET_ERROR_MSG(msg,                                                     \
                    "cuSparse error encountered at: ",                       \
                    "call='%s', Reason=%d:%s",                               \
                    #call,                                                   \
                    status,                                                  \
                    raft::sparse::detail::cusparse_error_to_string(status)); \
    }                                                                        \
  } while (0)

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct cuopt_cusparse_dnvector_t {
  cusparseDnVecDescr_t vec_descr{nullptr};

  cuopt_cusparse_dnvector_t() = default;

  cuopt_cusparse_dnvector_t(cuopt_cusparse_dnvector_t const& other)            = delete;
  cuopt_cusparse_dnvector_t& operator=(cuopt_cusparse_dnvector_t const& other) = delete;

  cuopt_cusparse_dnvector_t& operator=(cuopt_cusparse_dnvector_t&& other)
  {
    if (vec_descr != nullptr) { RAFT_CUSPARSE_TRY(cusparseDestroyDnVec(vec_descr)); }
    vec_descr       = other.vec_descr;
    other.vec_descr = nullptr;
    return *this;
  }

  cuopt_cusparse_dnvector_t(const rmm::device_uvector<f_t>& vec)
  {
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednvec(
      &vec_descr, vec.size(), const_cast<f_t*>(vec.data())));
  }

  ~cuopt_cusparse_dnvector_t()
  {
    if (vec_descr != nullptr) { CUOPT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnVec(vec_descr)); }
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
