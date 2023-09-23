// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 00:53:54 on Mon, Feb 13, 2023
//
// Description: wmma base hgemm

#include "common.h"

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 8   // BLOCK_COLS / WMMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / WMMA_M

#define WARP_ROW_TILES 4  // WARP_COLS / WMMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / WMMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / WMMA_K

#define CHUNK_LINE_BYTES 64          // CHUNK_K * WMMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * sizeof(int4) / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * WMMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

using namespace nvcuda;

__global__ void wmmaBaseKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, WMMA_M);
    const size_t N_tiles = div_ceil(N, WMMA_N);
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t block_tile_i =
        (blockIdx.z % 2) ? ((gridDim.y - blockIdx.y - 1) * BLOCK_COL_TILES) : (blockIdx.y * BLOCK_COL_TILES);
    const size_t block_tile_j = (blockIdx.z * gridDim.x + blockIdx.x) * BLOCK_ROW_TILES;

    if (block_tile_i >= M_tiles || block_tile_j >= N_tiles) {
        return;
    }

    extern __shared__ half smem[][AB_SMEM_STRIDE];

    const size_t warp_id = threadIdx.x / WARP_SIZE;
    const size_t lane_id = threadIdx.x % WARP_SIZE;

    constexpr size_t B_smem_idx_off = BLOCK_ROWS;

    half *smem_warp_tile_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS +
                               (warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET;

    half *smem_warp_stream_ptr = &smem[0][0] + warp_id * WMMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * WMMA_M * N + block_tile_j * WMMA_N;
    half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag[WARP_COL_TILES][WARP_ROW_TILES];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            wmma::fill_fragment(C_frag[i][j], 0.0);
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * WMMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * WMMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

#pragma unroll
    for (size_t tile_k = 0; tile_k < K_tiles; tile_k += CHUNK_K) {
        size_t A_smem_idx = BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        int4 *A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters; ++i) {
            *((int4 *)&smem[A_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *A_lane_ptr;

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        size_t B_smem_idx = B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        int4 *B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * WMMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                           (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters; ++i) {
            *((int4 *)&smem[B_smem_idx][0] + (lane_id % CHUNK_COPY_LINE_LANES)) = *B_lane_ptr;

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

#pragma unroll
        for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[WARP_COL_TILES];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[WARP_ROW_TILES];

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
                size_t A_smem_idx = (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * WMMA_M;
                const half *A_tile_ptr = &smem[A_smem_idx][k_step * WMMA_K];

                wmma::load_matrix_sync(A_frag[i], A_tile_ptr, WMMA_K * CHUNK_K);
            }

#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t B_smem_idx = B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * WMMA_N;
                const half *B_tile_ptr = &smem[B_smem_idx][k_step * WMMA_K];

                wmma::load_matrix_sync(B_frag[j], B_tile_ptr, WMMA_K * CHUNK_K);
            }

#pragma unroll
            for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
                for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                    wmma::mma_sync(C_frag[i][j_s], A_frag[i], B_frag[j_s], C_frag[i][j_s]);
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *C_tile_ptr = smem_warp_tile_ptr + i * C_SMEM_STRIDE * WMMA_M + j * WMMA_N;

            wmma::store_matrix_sync(C_tile_ptr, C_frag[i][j], C_SMEM_STRIDE, wmma::mem_row_major);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WMMA_M; ++i) {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) + lane_id % 16);
    }
}

size_t initWmmaBase() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size =
        std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half), BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(wmmaBaseKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void wmmaBase(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initWmmaBase();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    wmmaBaseKernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}
