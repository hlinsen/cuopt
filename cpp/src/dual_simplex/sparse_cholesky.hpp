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

#include "dual_simplex/dense_vector.hpp"
#include "dual_simplex/sparse_matrix.hpp"
#include "dual_simplex/simplex_solver_settings.hpp"
#include "dual_simplex/tic_toc.hpp"


#include <cuda_runtime.h>

#include "cholmod.h"
#include "cudss.h"

namespace cuopt::linear_programming::dual_simplex {


template <typename i_t, typename f_t>
class sparse_cholesky_base_t {
 public:
    virtual ~sparse_cholesky_base_t() = default;
    virtual i_t analyze(const csc_matrix_t<i_t, f_t>& A_in) = 0;
    virtual i_t factorize(const csc_matrix_t<i_t, f_t>& A_in) = 0;
    virtual i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) = 0;
    virtual void set_positive_definite(bool positive_definite) = 0;
};


template <typename i_t, typename f_t>
class sparse_cholesky_cholmod_t : public sparse_cholesky_base_t<i_t, f_t> {
 public:
    sparse_cholesky_cholmod_t(const simplex_solver_settings_t<i_t, f_t>& settings, i_t size) : n(size), first_factor(true) {
        cholmod_start(&common);
        int version[3];
        cholmod_version(version);
        settings.log.printf("Using CHOLMOD %d.%d.%d\n", version[0], version[1], version[2]);
        A = nullptr;
        L = nullptr;
    }
    ~sparse_cholesky_cholmod_t() override{
        cholmod_free_factor(&L, &common);
        cholmod_free_sparse(&A, &common);
        cholmod_finish(&common);
    }
    i_t analyze(const csc_matrix_t<i_t, f_t>& A_in) override
    {
        A = to_cholmod(A_in);
        // Perform symbolic analysis
        f_t start_symbolic = tic();
        common.nmethods = 1;
        common.method[0].ordering = CHOLMOD_AMD;
        L = cholmod_analyze(A, &common);
        f_t symbolic_time = toc(start_symbolic);
        printf("Symbolic method used %d\n", L->ordering);
        printf("Symbolic time %.2fs\n", symbolic_time);
        printf("Symbolic nonzeros in factor %e\n", common.method[common.selected].lnz);
        printf("Symbolic flops %e\n", common.method[common.selected].fl);
        return 0;
    }
    i_t factorize(const csc_matrix_t<i_t, f_t>& A_in) override
    {
        cholmod_free_sparse(&A, &common);
        A = to_cholmod(A_in);
        f_t start_numeric = tic();
        cholmod_factorize(A, L, &common);
        f_t numeric_time = toc(start_numeric);
        if (first_factor) {
            printf("Factor nonzeros %e\n", common.method[common.selected].lnz);
            printf("Factor time %.2fs\n", numeric_time);
            first_factor = false;
        }
        if (common.status < CHOLMOD_OK) {
            printf("Factorization failed\n");
            exit(1);
        }
        if (((int32_t) L->minor) != A_in.m) {
            printf("AA' not positive definite %ld minors versus %d\n", L->minor, A_in.m);
            return -1;
        }
        return 0;
    }
    i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) override
    {
        cholmod_dense *b_cholmod = cholmod_zeros(n, 1, CHOLMOD_REAL, &common);
        for (i_t i = 0; i < n; i++) {
            ((float64_t *)b_cholmod->x)[i] = b[i];
        }
        cholmod_dense *x_cholmod = cholmod_solve(CHOLMOD_A, L, b_cholmod, &common);
        for (i_t i = 0; i < n; i++) {
            x[i] = ((float64_t *)x_cholmod->x)[i];
        }

#ifdef CHECK_SOLVE
        int32_t no_transpose = 0;
        float64_t alpha[2] = {1.0, 0.0};
        float64_t beta[2] = {0.0, 0.0};
        cholmod_dense *residual = cholmod_zeros(n, 1, CHOLMOD_REAL, &common);

        cholmod_sdmult(A, no_transpose, alpha, beta, x_cholmod, residual, &common);
        for (i_t i = 0; i < n; i++) {
            f_t err = std::abs(((float64_t *)residual->x)[i] - b[i]);
            if (err > 1e-6) {
                printf("Error: L*L'*x[%d] - b[%d] = %e, x[%d] = %e, b[%d] = %e cholmod_b[%d] = %e\n", i, i, err, i, x[i], i, b[i], i, ((float64_t *)b_cholmod->x)[i]);
            }
        }
        beta[0] = -1.0;
        cholmod_sdmult(A, no_transpose, alpha, beta, x_cholmod, b_cholmod, &common);


        printf("|| L*L'*x - b || = %e\n", cholmod_norm_dense(b_cholmod, 0, &common));
#endif

        cholmod_free_dense(&x_cholmod, &common);
        cholmod_free_dense(&b_cholmod, &common);
        return 0;
    }

    void set_positive_definite(bool positive_definite) override
    {
        // Do nothing
    }

  private:

    cholmod_sparse *to_cholmod(const csc_matrix_t<i_t, f_t>& A_in)
    {
        i_t nnz = A_in.col_start[A_in.n];
        cholmod_sparse *A = cholmod_allocate_sparse(A_in.m, A_in.n, nnz, 0, 0, 0, CHOLMOD_REAL, &common);
        for (i_t j = 0; j <= A_in.n; j++) {
            ((int32_t *)A->p)[j] = A_in.col_start[j];
        }
        for (i_t j = 0; j < A_in.n; j++) {
            ((int32_t *)A->nz)[j] = A_in.col_start[j + 1] - A_in.col_start[j];
        }
        for (i_t p = 0; p < nnz; p++) {
            ((int32_t *)A->i)[p] = A_in.i[p];
        }
        for (i_t p = 0; p < nnz; p++) {
            ((float64_t *)A->x)[p] = A_in.x[p];
        }
        A->nzmax = nnz;
        A->stype = 1;
        cholmod_check_sparse(A, &common);
        return A;
    }
    i_t n;
    bool first_factor;
    cholmod_sparse *A;
    cholmod_factor *L;
    cholmod_common common;
};

#define CUDSS_EXAMPLE_FREE do {} while(0)

#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE; \
            return -1; \
        } \
    } while(0);

#define CUDA_CALL_AND_CHECK_EXIT(call, msg)                                                 \
  do {                                                                                 \
    cuda_error = call;                                                                 \
    if (cuda_error != cudaSuccess) {                                                   \
      printf("FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
      CUDSS_EXAMPLE_FREE;                                                              \
      exit(-1);                                                                       \
    }                                                                                  \
  } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -2; \
        } \
    } while(0);

#define CUDSS_CALL_AND_CHECK_EXIT(call, status, msg)                                          \
  do {                                                                                        \
    status = call;                                                                            \
    if (status != CUDSS_STATUS_SUCCESS) {                                                     \
      printf("FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", \
             status);                                                                         \
      CUDSS_EXAMPLE_FREE;                                                                     \
      exit(-2);                                                                              \
    }                                                                                         \
  } while (0);

template <typename i_t, typename f_t>
class sparse_cholesky_cudss_t : public sparse_cholesky_base_t<i_t, f_t> {
 public:
    sparse_cholesky_cudss_t(const simplex_solver_settings_t<i_t, f_t>& settings, i_t size) : n(size), nnz(-1), first_factor(true), positive_definite(true) {

        int major, minor, patch;
        cudssGetProperty(MAJOR_VERSION, &major);
        cudssGetProperty(MINOR_VERSION, &minor);
        cudssGetProperty(PATCH_LEVEL,   &patch);
        settings.log.printf("CUDSS Version %d.%d.%d\n", major, minor, patch);

        cuda_error = cudaSuccess;
        status = CUDSS_STATUS_SUCCESS;
        CUDA_CALL_AND_CHECK_EXIT(cudaStreamCreate(&stream), "cudaStreamCreate");
        CUDSS_CALL_AND_CHECK_EXIT(cudssCreate(&handle), status, "cudssCreate");
        CUDSS_CALL_AND_CHECK_EXIT(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
        CUDSS_CALL_AND_CHECK_EXIT(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

#ifdef USE_AMD
        // Tell cuDSS to use AMD
        cudssAlgType_t reorder_alg = CUDSS_ALG_3;
        CUDSS_CALL_AND_CHECK_EXIT(cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG,
                         &reorder_alg, sizeof(cudssAlgType_t)), status, "cudssConfigSet for reordering alg");
#endif

        int32_t ir_n_steps = 2;
        CUDSS_CALL_AND_CHECK_EXIT(cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS,
                                          &ir_n_steps, sizeof(int32_t)), status, "cudssConfigSet for ir n steps");

        // Device pointers
        csr_offset_d = nullptr;
        csr_columns_d = nullptr;
        csr_values_d = nullptr;
        x_values_d = nullptr;
        b_values_d = nullptr;
        CUDA_CALL_AND_CHECK_EXIT(cudaMalloc(&x_values_d, n * sizeof(f_t)), "cudaMalloc for x_values");
        CUDA_CALL_AND_CHECK_EXIT(cudaMalloc(&b_values_d, n * sizeof(f_t)), "cudaMalloc for b_values");

        i_t ldb = n;
        i_t ldx = n;
        CUDSS_CALL_AND_CHECK_EXIT(
          cudssMatrixCreateDn(&cudss_b, n, 1, ldb, b_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          status,
          "cudssMatrixCreateDn for b");
        CUDSS_CALL_AND_CHECK_EXIT(
          cudssMatrixCreateDn(&cudss_x, n, 1, ldx, x_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          status,
          "cudssMatrixCreateDn for x");
    }
    ~sparse_cholesky_cudss_t() override {
        cudaFree(csr_values_d);
        cudaFree(csr_columns_d);
        cudaFree(csr_offset_d);

        cudaFree(x_values_d);
        cudaFree(b_values_d);

        CUDSS_CALL_AND_CHECK_EXIT(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");


        CUDSS_CALL_AND_CHECK_EXIT(cudssMatrixDestroy(cudss_x), status, "cudssMatrixDestroy for cudss_x");
        CUDSS_CALL_AND_CHECK_EXIT(cudssMatrixDestroy(cudss_b), status, "cudssMatrixDestroy for cudss_b");
        CUDSS_CALL_AND_CHECK_EXIT(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
        CUDSS_CALL_AND_CHECK_EXIT(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
        CUDSS_CALL_AND_CHECK_EXIT(cudssDestroy(handle), status, "cudssDestroy");
        CUDA_CALL_AND_CHECK_EXIT(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        CUDA_CALL_AND_CHECK_EXIT(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }
    i_t analyze(const csc_matrix_t<i_t, f_t>& A_in) override
    {
        csr_matrix_t<i_t, f_t> Arow;
        A_in.to_compressed_row(Arow);

        nnz = A_in.col_start[A_in.n];

        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offset_d, (n + 1) * sizeof(i_t)), "cudaMalloc for csr_offset");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(i_t)), "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(f_t)), "cudaMalloc for csr_values");

        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offset_d, Arow.row_start.data(), (n + 1) * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy for csr_offset");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, Arow.j.data(), nnz * sizeof(i_t), cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, Arow.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");

        if (!first_factor) {
            CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
        }

        CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A,
                                                  n,
                                                  n,
                                                  nnz,
                                                  csr_offset_d,
                                                  nullptr,
                                                  csr_columns_d,
                                                  csr_values_d,
                                                  CUDA_R_32I,
                                                  CUDA_R_64F,
                                                  positive_definite ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_SYMMETRIC,
                                                  CUDSS_MVIEW_FULL,
                                                  CUDSS_BASE_ZERO),
                             status,
                             "cudssMatrixCreateCsr");

        // Perform symbolic analysis
        f_t start_symbolic = tic();

        CUDSS_CALL_AND_CHECK(
          cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, cudss_x, cudss_b),
          status,
          "cudssExecute for analysis");

        f_t symbolic_time = toc(start_symbolic);
        printf("Symbolic time %.2fs\n", symbolic_time);
        int64_t lu_nz       = 0;
        size_t size_written = 0;
        CUDSS_CALL_AND_CHECK(
          cudssDataGet(
            handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nz, sizeof(int64_t), &size_written),
          status,
          "cudssDataGet for LU_NNZ");
        printf("Symbolic nonzeros in factor %e\n", static_cast<f_t>(lu_nz) / 2.0);
        // TODO: Is there any way to get nonzeros in the factors?
        // TODO: Is there any way to get flops for the factorization?


        return 0;
    }
    i_t factorize(const csc_matrix_t<i_t, f_t>& A_in) override
    {

        csr_matrix_t<i_t, f_t> Arow;
        A_in.to_compressed_row(Arow);

        if (nnz != A_in.col_start[A_in.n]) {
            printf("Error: nnz %d != A_in.col_start[A_in.n] %d\n", nnz, A_in.col_start[A_in.n]);
            exit(1);
        }

        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, Arow.x.data(), nnz * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");

        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(A, csr_values_d), status, "cudssMatrixSetValues for A");

        f_t start_numeric = tic();
        CUDSS_CALL_AND_CHECK(
          cudssExecute(
            handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, cudss_x, cudss_b),
          status,
          "cudssExecute for factorization");

        f_t numeric_time = toc(start_numeric);

        int info;
        size_t sizeWritten = 0;
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_INFO, &info,
                             sizeof(info), &sizeWritten), status, "cudssDataGet for info");
        if (info != 0) {
            printf("Factorization failed info %d\n", info);
            return -1;
        }

        if (first_factor) {
            printf("Factor time %.2fs\n", numeric_time);
            first_factor = false;
        }
        if (status != CUDSS_STATUS_SUCCESS) {
            printf("cuDSS Factorization failed\n");
            return -1;
        }
        return 0;
    }

    i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) override
    {
        if (b.size() != n) {
            printf("Error: b.size() %d != n %d\n", b.size(), n);
            exit(1);
        }
        if (x.size() != n) {
            printf("Error: x.size() %d != n %d\n", x.size(), n);
            exit(1);
        }
        CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b.data(), n * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_d, x.data(), n * sizeof(f_t), cudaMemcpyHostToDevice), "cudaMemcpy for x_values");

        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(cudss_b, b_values_d), status, "cudssMatrixSetValues for b");
        CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(cudss_x, x_values_d), status, "cudssMatrixSetValues for x");


        i_t ldb = n;
        i_t ldx = n;
        CUDSS_CALL_AND_CHECK_EXIT(
          cudssMatrixCreateDn(&cudss_b, n, 1, ldb, b_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          status,
          "cudssMatrixCreateDn for b");
        CUDSS_CALL_AND_CHECK_EXIT(
          cudssMatrixCreateDn(&cudss_x, n, 1, ldx, x_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          status,
          "cudssMatrixCreateDn for x");

        CUDSS_CALL_AND_CHECK(
          cudssExecute(
            handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, cudss_x, cudss_b),
          status,
          "cudssExecute for solve");

        CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CALL_AND_CHECK(cudaMemcpy(x.data(), x_values_d, n * sizeof(f_t), cudaMemcpyDeviceToHost), "cudaMemcpy for x");

        for (i_t i = 0; i < n; i++) {
            if (x[i] != x[i]) {
                return -1;
            }
        }

        return 0;
    }

    void set_positive_definite(bool positive_definite) override
    {
        this->positive_definite = positive_definite;
    }

  private:


    i_t n;
    i_t nnz;
    bool first_factor;
    bool positive_definite;
    cudaError_t cuda_error;
    cudssStatus_t status;
    cudaStream_t stream;
    cudssHandle_t handle;
    cudssConfig_t solverConfig;
    cudssData_t solverData;
    cudssMatrix_t A;
    cudssMatrix_t cudss_x;
    cudssMatrix_t cudss_b;
    i_t *csr_offset_d;
    i_t *csr_columns_d;
    f_t *csr_values_d;
    f_t *x_values_d;
    f_t *b_values_d;
};


} // namespace cuopt::linear_programming::dual_simplex
