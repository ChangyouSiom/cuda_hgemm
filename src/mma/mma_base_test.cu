#include "common.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define THREADS_PER_BLOCK 256
#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define M_TILES 16  // BLOCK_ROWS/MMA_M
#define N_TILES 16  // BLOCK_COLS/MMA_N

#define CHUNK 2

#define AB_SMEM_STRIDE 32  // CHUNK * MMA_K
#define C_SMEM_STRIDE 128  // BLOCK_COLS

#define WARPS_PER_BLOCK 8  // THREADS_PER_BLOCK/32

#define WARP_TILES 32  // M_TILES*NTILES/WARPS_PER_BLOCK
#define WARP_COL_TILES 4
#define WARP_ROW_TILES 8  // WARP_TILES/WARP_COL_TILES

__global__ void mmaBaseKernelTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    extern __shared__ half smem[][AB_SMEM_STRIDE];

    uint32_t RC[WARP_COL_TILES][WARP_ROW_TILES][2] = {0};

    int k_tiles = K / MMA_K;
#pragma unroll
    for (int tile_k = 0; tile_k < k_tiles; tile_k += CHUNK) {
// A : global to share
#pragma unroll
        for (int i = 0; i < BLOCK_ROWS / WARPS_PER_BLOCK / 8; ++i) {
            int shared_row = warp_id * BLOCK_ROWS / WARPS_PER_BLOCK + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_row = blockIdx.y * BLOCK_ROWS + shared_row;
            int block_col = tile_k * MMA_K + shared_col;
            *((int4 *)(smem[shared_row] + shared_col)) = *((int4 *)(A + K * block_row + block_col));
        }
// B : global to share
#pragma unroll
        for (int i = 0; i < BLOCK_COLS / WARPS_PER_BLOCK / 8; ++i) {
            int shared_row = warp_id * BLOCK_COLS / WARPS_PER_BLOCK + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_col = blockIdx.x * BLOCK_COLS + shared_row;
            int block_row = tile_k * MMA_K + shared_col;
            *((int4 *)(smem[BLOCK_ROWS + shared_row] + shared_col)) = *((int4 *)(B + K * block_col + block_row));
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            uint32_t RA[WARP_COL_TILES][4];
            uint32_t RB[WARP_ROW_TILES][2];
#pragma unroll
            for (int warp_tile_m = 0; warp_tile_m < WARP_COL_TILES; warp_tile_m++) {
                int a_row = warp_id / 2 * (MMA_M * WARP_COL_TILES) + warp_tile_m * MMA_M;
                int a_col = k * MMA_K;
                uint32_t a_smem_lane_addr =
                    __cvta_generic_to_shared(&smem[a_row + lane_id % 16][a_col + (lane_id / 16) * 8]);
                LDMATRIX_X4(RA[warp_tile_m][0], RA[warp_tile_m][1], RA[warp_tile_m][2], RA[warp_tile_m][3],
                            a_smem_lane_addr);
            }

#pragma unroll
            for (int warp_tile_n = 0; warp_tile_n < WARP_ROW_TILES; warp_tile_n++) {
                int b_row = k * MMA_K;
                int b_col = warp_id % 2 * (MMA_N * WARP_ROW_TILES) + warp_tile_n * MMA_N;
                uint32_t b_smem_lane_addr =
                    __cvta_generic_to_shared(&smem[BLOCK_ROWS + b_col + lane_id % 8][b_row + ((lane_id / 8) % 2) * 8]);
                LDMATRIX_X2(RB[warp_tile_n][0], RB[warp_tile_n][1], b_smem_lane_addr);
            }

#pragma unroll
            for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
                for (int j = 0; j < WARP_ROW_TILES; j++) {
                    size_t j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j;
                    HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[i][0],
                              RA[i][1], RA[i][2], RA[i][3], RB[j_s][0], RB[j_s][1],
                              RC[i][j_s][0], RC[i][j_s][1]);
                }
            }
        }
        // __syncthreads();  取消这个块内同步同步看起来没有问题。这个同步，本质上是warp之间发生了协作。而这里并没有协作，所以可以取消。
    }
// register to shared
#pragma unroll
    for (int warp_tile_m = 0; warp_tile_m < WARP_COL_TILES; warp_tile_m++) {
#pragma unroll
        for (int warp_tile_n = 0; warp_tile_n < WARP_ROW_TILES; warp_tile_n++) {
            int c_row = warp_id / 2 * (MMA_M * WARP_COL_TILES) + warp_tile_m * MMA_M;
            int c_col = warp_id % 2 * (MMA_N * WARP_ROW_TILES) + warp_tile_n * MMA_N;
            *((uint32_t *)(&smem[0][0] + (c_row + lane_id / 4) * C_SMEM_STRIDE + c_col) + lane_id % 4) =
                RC[warp_tile_m][warp_tile_n][0];
            *((uint32_t *)(&smem[0][0] + (c_row + lane_id / 4 + 8) * C_SMEM_STRIDE + c_col) + lane_id % 4) =
                RC[warp_tile_m][warp_tile_n][1];
        }
    }
    __syncthreads();
// C shared to global
#pragma unroll
    for (int i = 0; i < BLOCK_ROWS / WARPS_PER_BLOCK / 2; ++i) {
        int shared_row = warp_id * BLOCK_ROWS / WARPS_PER_BLOCK + lane_id / 16 + i * 2;
        int shared_col = lane_id % 16 * 8;
        int block_row = blockIdx.y * BLOCK_ROWS + shared_row;
        int block_col = blockIdx.x * BLOCK_COLS + shared_col;
        *((int4 *)(C + N * block_row + block_col)) =
            *((int4 *)(&smem[0][0] + shared_row * C_SMEM_STRIDE + shared_col));
    }
}

size_t initMmaBaseTest() {
    int dev_id = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDevice(&dev_id));
    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, dev_id));
    size_t ab_shared_size = (BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(half);
    size_t c_shared_size = BLOCK_ROWS * C_SMEM_STRIDE * sizeof(half);
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