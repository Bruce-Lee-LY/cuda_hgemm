// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:02:28 on Tue, Feb 28, 2023
//
// Description: mma async stage4 hgemm

#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define WARP_ROWS 64
#define WARP_COLS 64

#define BLOCK_ROW_WARPS 2  // BLOCK_COLS / WARP_COLS
#define BLOCK_COL_WARPS 4  // BLOCK_ROWS / WARP_ROWS

#define BLOCK_ROW_TILES 16  // BLOCK_COLS / MMA_N
#define BLOCK_COL_TILES 16  // BLOCK_ROWS / MMA_M

#define WARP_ROW_TILES 8  // WARP_COLS / MMA_N
#define WARP_COL_TILES 4  // WARP_ROWS / MMA_M

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8      // BLOCK_ROW_WARPS * BLOCK_COL_WARPS
#define THREADS_PER_BLOCK 256  // WARP_SIZE * WARPS_PER_BLOCK

#define CHUNK_K 2  // 32 / MMA_K

#define THREAD_COPY_BYTES 16

#define CHUNK_LINE_BYTES 64          // CHUNK_K * MMA_K * sizeof(half)
#define CHUNK_COPY_LINES_PER_WARP 8  // WARP_SIZE * THREAD_COPY_BYTES / CHUNK_LINE_BYTES
#define CHUNK_COPY_LINE_LANES 4      // WARP_SIZE / CHUNK_COPY_LINES_PER_WARP

#define AB_SMEM_STRIDE 32  // CHUNK_K * MMA_K

#define C_SMEM_STRIDE 128  // BLOCK_COLS
#define C_SMEM_OFFSET 64   // WARP_COLS

#define BLOCK_STRIDE 16

#define SMEM_BANK_ROWS 2  // 32 * 4 / (AB_SMEM_STRIDE * sizeof(half))

#define PERMUTED_OFFSET 8
#define PERMUTED_COLS 4

#define K_STAGE 4

__global__ void mmaAsyncStage4Kernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                     size_t M, size_t N, size_t K) {
    const size_t M_tiles = div_ceil(M, MMA_M);
    const size_t N_tiles = div_ceil(N, MMA_N);
    const size_t K_tiles = div_ceil(K, MMA_K);

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
    constexpr size_t smem_stage_off = BLOCK_ROWS + BLOCK_COLS;

    half *smem_warp_tile_row_ptr = &smem[0][0] + (warp_id / BLOCK_ROW_WARPS) * C_SMEM_STRIDE * WARP_ROWS;
    const half *smem_warp_stream_ptr = &smem[0][0] + warp_id * MMA_M * 2 * C_SMEM_STRIDE;

    const size_t gmem_idx = (block_tile_i + warp_id * 2) * MMA_M * N + block_tile_j * MMA_N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2];

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            RC[i][j][0] = 0;
            RC[i][j][1] = 0;
        }
    }

    const half *A_warp_ptr = &A[block_tile_i * MMA_M * K] + BLOCK_ROWS / WARPS_PER_BLOCK * K * warp_id;
    const half *B_warp_ptr = &B[block_tile_j * MMA_N * K] + BLOCK_COLS / WARPS_PER_BLOCK * K * warp_id;

    constexpr size_t A_smem_iters = BLOCK_ROWS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);
    constexpr size_t B_smem_iters = BLOCK_COLS / (CHUNK_COPY_LINES_PER_WARP * WARPS_PER_BLOCK);

    size_t smem_store_idx = 0;
    size_t smem_load_idx = 0;

    size_t smem_store_off = 0;
    size_t smem_load_off = 0;

    size_t A_smem_idx = 0;
    int4 *A_lane_ptr = nullptr;

    size_t B_smem_idx = 0;
    int4 *B_lane_ptr = nullptr;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + (lane_id / CHUNK_COPY_LINE_LANES) * K) + (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();

    smem_store_idx = (smem_store_idx + 1) % K_STAGE;
    smem_store_off = smem_store_idx * smem_stage_off;

    A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
    A_lane_ptr = (int4 *)(A_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < A_smem_iters; ++i) {
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

        A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
    B_lane_ptr = (int4 *)(B_warp_ptr + 2 * CHUNK_K * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                 (lane_id % CHUNK_COPY_LINE_LANES);
    B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
    for (size_t i = 0; i < B_smem_iters; ++i) {
        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                    ((lane_id % CHUNK_COPY_LINE_LANES +
                                      (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                     CHUNK_COPY_LINE_LANES) *
                                        THREAD_COPY_BYTES;

        CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

        B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
        B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
    }

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(2);

    __syncthreads();

    uint32_t RA[2][WARP_COL_TILES][4];
    uint32_t RB[2][WARP_ROW_TILES][2];

    size_t reg_store_idx = 0;
    size_t reg_load_idx = 1;

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
        size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
            &smem[A_smem_idx + lane_id % 16][((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                       SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                             AB_SMEM_STRIDE]);

        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    A_smem_lane_addr);
    }

#pragma unroll
    for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
        size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
        uint32_t B_smem_lane_addr =
            __cvta_generic_to_shared(&smem[B_smem_idx + lane_id % 8]
                                          [(((lane_id / 8) % 2) * 8 + (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                          SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                           AB_SMEM_STRIDE]);

        LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
    }

#pragma unroll
    for (size_t tile_k = CHUNK_K * (K_STAGE - 1); tile_k < K_tiles; tile_k += CHUNK_K) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        smem_store_idx = (smem_store_idx + 1) % K_STAGE;
        smem_store_off = smem_store_idx * smem_stage_off;

        A_smem_idx = smem_store_off + BLOCK_ROWS / WARPS_PER_BLOCK * warp_id;
        A_lane_ptr = (int4 *)(A_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        A_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < A_smem_iters / CHUNK_K; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        B_smem_idx = smem_store_off + B_smem_idx_off + BLOCK_COLS / WARPS_PER_BLOCK * warp_id;
        B_lane_ptr = (int4 *)(B_warp_ptr + tile_k * MMA_K + (lane_id / CHUNK_COPY_LINE_LANES) * K) +
                     (lane_id % CHUNK_COPY_LINE_LANES);
        B_smem_idx += lane_id / CHUNK_COPY_LINE_LANES;

#pragma unroll
        for (size_t i = 0; i < B_smem_iters / CHUNK_K; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        smem_load_idx = (smem_load_idx + 1) % K_STAGE;
        smem_load_off = smem_load_idx * smem_stage_off;

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * A_smem_iters / CHUNK_K; i < A_smem_iters; ++i) {
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&smem[A_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (A_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(A_smem_lane_addr, A_lane_ptr, THREAD_COPY_BYTES);

            A_lane_ptr = (int4 *)((half *)A_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            A_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

#pragma unroll
        for (size_t i = (CHUNK_K - 1) * B_smem_iters / CHUNK_K; i < B_smem_iters; ++i) {
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&smem[B_smem_idx][0]) +
                                        ((lane_id % CHUNK_COPY_LINE_LANES +
                                          (B_smem_idx % (CHUNK_COPY_LINE_LANES * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) %
                                         CHUNK_COPY_LINE_LANES) *
                                            THREAD_COPY_BYTES;

            CP_ASYNC_CG(B_smem_lane_addr, B_lane_ptr, THREAD_COPY_BYTES);

            B_lane_ptr = (int4 *)((half *)B_lane_ptr + CHUNK_COPY_LINES_PER_WARP * K);
            B_smem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(2);

        __syncthreads();

        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr =
                __cvta_generic_to_shared(&smem[A_smem_idx + lane_id % 16]
                                              [((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) /
                                                                         SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                                               AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
    }

#pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(((k_step + 1) % CHUNK_K) * MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(((k_step + 1) % CHUNK_K) * MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;

            CP_ASYNC_WAIT_GROUP(1);

            __syncthreads();
        }
    }

#pragma unroll
    for (size_t k_step = 0; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(((k_step + 1) % CHUNK_K) * MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(((k_step + 1) % CHUNK_K) * MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }

        if (k_step + 2 == CHUNK_K) {
            smem_load_idx = (smem_load_idx + 1) % K_STAGE;
            smem_load_off = smem_load_idx * smem_stage_off;

            CP_ASYNC_WAIT_GROUP(0);

            __syncthreads();
        }
    }

#pragma unroll
    for (size_t k_step = 1; k_step < CHUNK_K; ++k_step) {
        reg_store_idx ^= 1;
        reg_load_idx ^= 1;

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
            size_t A_smem_idx = smem_load_off + (warp_id / BLOCK_ROW_WARPS) * WARP_ROWS + i * MMA_M;
            uint32_t A_smem_lane_addr = __cvta_generic_to_shared(
                &smem[A_smem_idx + lane_id % 16]
                     [(k_step * MMA_K + (lane_id / 16) * 8 +
                       (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1], RA[reg_store_idx][i][2],
                        RA[reg_store_idx][i][3], A_smem_lane_addr);
        }

#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t B_smem_idx = smem_load_off + B_smem_idx_off + (warp_id % BLOCK_ROW_WARPS) * WARP_COLS + j * MMA_N;
            uint32_t B_smem_lane_addr = __cvta_generic_to_shared(
                &smem[B_smem_idx + lane_id % 8]
                     [(k_step * MMA_K + ((lane_id / 8) % 2) * 8 +
                       (lane_id % 8 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) %
                      AB_SMEM_STRIDE]);

            LDMATRIX_X2(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1], B_smem_lane_addr);
        }

#pragma unroll
        for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
            for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
                size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

                HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0], RA[reg_load_idx][i][1],
                          RA[reg_load_idx][i][2], RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                          RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;

            HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                      RA[reg_store_idx][i][2], RA[reg_store_idx][i][3], RB[reg_store_idx][j_s][0],
                      RB[reg_store_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < WARP_COL_TILES; ++i) {
#pragma unroll
        for (size_t j = 0; j < WARP_ROW_TILES; ++j) {
            half *lane_ptr0 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;
            half *lane_ptr1 =
                smem_warp_tile_row_ptr + (i * MMA_M + lane_id / 4 + 8) * C_SMEM_STRIDE +
                ((warp_id % BLOCK_ROW_WARPS) * C_SMEM_OFFSET + j * MMA_N +
                 (lane_id % 4) * sizeof(uint32_t) / sizeof(half) + ((lane_id / 4 + 8) % 8) * PERMUTED_OFFSET) %
                    C_SMEM_STRIDE;

            *((uint32_t *)(lane_ptr0)) = RC[i][j][0];
            *((uint32_t *)(lane_ptr1)) = RC[i][j][1];
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = 0; i < MMA_M; ++i) {
        *((int4 *)(src_gmem_warp_stream_ptr + (i * 2 + lane_id / 16) * N) + lane_id % 16) =
            *((int4 *)(smem_warp_stream_ptr + (i * 2 + lane_id / 16) * C_SMEM_STRIDE) +
              (lane_id % 16 + (i * 2 + lane_id / 16) % 8) % (C_SMEM_STRIDE * sizeof(half) / THREAD_COPY_BYTES));
    }
}

size_t initMmaAsyncStage4() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));

    size_t smem_max_size = std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half) * K_STAGE,
                                    BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half));
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);

    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(mmaAsyncStage4Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));

    return smem_max_size;
}

void mmaAsyncStage4(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaAsyncStage4();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));

    mmaAsyncStage4Kernel<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}
