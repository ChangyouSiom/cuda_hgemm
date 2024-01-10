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
    int block_tile_i = blockIdx.x;
    int block_tile_j = blockIdx.y;

    int thread_id = threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;
    extern __shared__ half smem[][SHARED_STRIDE_AB];

    uint32_t RC[WARP_TILES_M][WARP_TILES_N][2] = {0};

    int k_tiles = K / WARP_K;
    for (int k_tile = 0; k_tile < k_tiles; k_tile += CHUNK) {
        // A : global to share
        for (int i = 0; i < BLOCK_ROWS / BLOCK_WARPS / 8; ++i) {
            int shared_row = warp_id * BLOCK_ROWS / BLOCK_WARPS + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_row = block_tile_i * BLOCK_ROWS + shared_row;
            int block_col = k_tile * CHUNK * WARP_K + shared_col;
            *((int4 *)(smem[shared_row] + shared_col)) = *((int4 *)(A + M * block_row + block_col));
        }
        // B : global to share
        for (int i = 0; i < BLOCK_COLS / BLOCK_WARPS / 8; ++i) {
            int shared_row = warp_id * BLOCK_COLS / BLOCK_WARPS + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_col = block_tile_j * BLOCK_COLS + shared_row;
            int block_row = k_tile * CHUNK * WARP_K + shared_col;
            *((int4 *)(smem[BLOCK_ROWS + shared_row] + shared_col)) = *((int4 *)(B + N * block_col + block_row));
        }
        __syncthreads();
        for (int k = 0; k < CHUNK; ++k) {
            for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
                for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
                    uint32_t RA[4];
                    uint32_t RB[2];
                    int a_row = warp_id / 2 * (WARP_M * WARP_TILES_M) + warp_tile_m * WARP_M;
                    int a_col = k * WARP_K;
                    int b_row = k * WARP_K;
                    int b_col = warp_id % 2 * (WARP_N * WARP_TILES_N) + warp_tile_n * WARP_N;
                    uint32_t a_smem_lane_addr =
                        __cvta_generic_to_shared(&smem[a_row + lane_id % 16][a_col + (lane_id / 16) * 8]);
                    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], a_smem_lane_addr);
                    uint32_t b_smem_lane_addr = __cvta_generic_to_shared(
                        &smem[BLOCK_ROWS + b_col + lane_id % 8][b_row + ((lane_id / 8) % 2) * 8]);
                    LDMATRIX_X2(RB[0], RB[1], b_smem_lane_addr);
                    HMMA16816(RC[warp_tile_m][warp_tile_n][0], RC[warp_tile_m][warp_tile_n][1], RA[0], RA[1], RA[2],
                              RA[3], RB[0], RB[1], RC[warp_tile_m][warp_tile_n][0], RC[warp_tile_m][warp_tile_n][1]);
                    __syncthreads();
                }
            }
        }
    }
    // register to shared

    // shared to global
}

size_t initMmaBaseTest() {
    size_t ab_shared_size = (BLOCK_ROWS + BLOCK_COLS) * SHARED_STRIDE_AB * sizeof(half);
    size_t c_shared_size = BLOCK_ROWS * SHARED_STRIDE_C * sizeof(half);
    return max(ab_shared_size, c_shared_size);
}

void mmaBaseTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaBaseTest();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, BLOCK_COLS), div_ceil(M, BLOCK_ROWS));

    mmaBaseKernelTest<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}