#include "common.h"

#define WARP_M 16
#define WARP_N 8
#define WARP_K 16

#define THREADS_PER_BLOCK 256
#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define M_TILES 16  // BLOCK_ROWS/WARP_M
#define N_TILES 16  // BLOCK_COLS/WARP_N

#define CHUNK 2

#define SHARED_STRIDE_AB 32  // CHUNK * WARP_K
#define SHARED_STRIDE_C 128  // BLOCK_COLS

#define BLOCK_WARPS 8  // THREADS_PER_BLOCK/32

#define WARP_TILES 32  // M_TILES*NTILES/BLOCK_WARPS
#define WARP_TILES_M 4
#define WARP_TILES_N 8  // WARP_TILES/WARP_TILES_M

__global__ void mmaBaseKernelTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    int block_tile_i = blockIdx.y;
    int block_tile_j = blockIdx.x;

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    extern __shared__ half smem[][SHARED_STRIDE_AB];

    uint32_t RC[WARP_TILES_M][WARP_TILES_N][2] = {0};

    int k_tiles = K / WARP_K;
#pragma unroll
    for (int k_tile = 0; k_tile < k_tiles; k_tile += CHUNK) {
// A : global to share
#pragma unroll
        for (int i = 0; i < BLOCK_ROWS / BLOCK_WARPS / 8; ++i) {
            int shared_row = warp_id * BLOCK_ROWS / BLOCK_WARPS + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_row = block_tile_i * BLOCK_ROWS + shared_row;
            int block_col = k_tile * WARP_K + shared_col;
            *((int4 *)(smem[shared_row] + shared_col)) = *((int4 *)(A + K * block_row + block_col));
        }
// B : global to share
#pragma unroll
        for (int i = 0; i < BLOCK_COLS / BLOCK_WARPS / 8; ++i) {
            int shared_row = warp_id * BLOCK_COLS / BLOCK_WARPS + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_col = block_tile_j * BLOCK_COLS + shared_row;
            int block_row = k_tile * WARP_K + shared_col;
            *((int4 *)(smem[BLOCK_ROWS + shared_row] + shared_col)) = *((int4 *)(B + K * block_col + block_row));
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            uint32_t RA[WARP_TILES_M][4];
            uint32_t RB[WARP_TILES_N][2];
#pragma unroll
            for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
                int a_row = warp_id / 2 * (WARP_M * WARP_TILES_M) + warp_tile_m * WARP_M;
                int a_col = k * WARP_K;
                uint32_t a_smem_lane_addr =
                    __cvta_generic_to_shared(&smem[a_row + lane_id % 16][a_col + (lane_id / 16) * 8]);
                LDMATRIX_X4(RA[warp_tile_m][0], RA[warp_tile_m][1], RA[warp_tile_m][2], RA[warp_tile_m][3],
                            a_smem_lane_addr);
            }

#pragma unroll
            for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
                int b_row = k * WARP_K;
                int b_col = warp_id % 2 * (WARP_N * WARP_TILES_N) + warp_tile_n * WARP_N;
                uint32_t b_smem_lane_addr =
                    __cvta_generic_to_shared(&smem[BLOCK_ROWS + b_col + lane_id % 8][b_row + ((lane_id / 8) % 2) * 8]);
                LDMATRIX_X2(RB[warp_tile_n][0], RB[warp_tile_n][1], b_smem_lane_addr);
            }

#pragma unroll
            for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
#pragma unroll
                for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
                    HMMA16816(RC[warp_tile_m][warp_tile_n][0], RC[warp_tile_m][warp_tile_n][1], RA[warp_tile_m][0],
                              RA[warp_tile_m][1], RA[warp_tile_m][2], RA[warp_tile_m][3], RB[warp_tile_n][0], RB[warp_tile_n][1],
                              RC[warp_tile_m][warp_tile_n][0], RC[warp_tile_m][warp_tile_n][1]);
                    __syncthreads();
                }
            }
        }
    }
// register to shared
#pragma unroll
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
#pragma unroll
        for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
            int c_row = warp_id / 2 * (WARP_M * WARP_TILES_M) + warp_tile_m * WARP_M;
            int c_col = warp_id % 2 * (WARP_N * WARP_TILES_N) + warp_tile_n * WARP_N;
            *((uint32_t *)(&smem[0][0] + (c_row + lane_id / 4) * SHARED_STRIDE_C + c_col) + lane_id % 4) =
                RC[warp_tile_m][warp_tile_n][0];
            *((uint32_t *)(&smem[0][0] + (c_row + lane_id / 4 + 8) * SHARED_STRIDE_C + c_col) + lane_id % 4) =
                RC[warp_tile_m][warp_tile_n][1];
        }
    }
    __syncthreads();
// C shared to global
#pragma unroll
    for (int i = 0; i < BLOCK_ROWS / BLOCK_WARPS / 2; ++i) {
        int shared_row = warp_id * BLOCK_ROWS / BLOCK_WARPS + lane_id / 16 + i * 2;
        int shared_col = lane_id % 16 * 8;
        int block_row = block_tile_i * BLOCK_ROWS + shared_row;
        int block_col = block_tile_j * BLOCK_COLS + shared_col;
        *((int4 *)(C + N * block_row + block_col)) =
            *((int4 *)(&smem[0][0] + shared_row * SHARED_STRIDE_C + shared_col));
    }
}

size_t initMmaBaseTest() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));
    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));
    size_t ab_shared_size = (BLOCK_ROWS + BLOCK_COLS) * SHARED_STRIDE_AB * sizeof(half);
    size_t c_shared_size = BLOCK_ROWS * SHARED_STRIDE_C * sizeof(half);
    size_t smem_max_size = std::max(ab_shared_size, c_shared_size);
    HLOG("smem_max_size: %.0f KBytes (%zu Bytes)", static_cast<double>(smem_max_size) / 1024, smem_max_size);
    HGEMM_CHECK_GT(dev_prop.sharedMemPerMultiprocessor, smem_max_size);
    HGEMM_CHECK_CUDART_ERROR(
        cudaFuncSetAttribute(mmaBaseKernelTest, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size));
    return smem_max_size;
}

void mmaBaseTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaBaseTest();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, BLOCK_COLS), div_ceil(M, BLOCK_ROWS));

    mmaBaseKernelTest<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}