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
    int wrap_id = thread_id / 32;
    int lane_id = thread_id % 32;
    extern __shared__ half smem[][SHARED_STRIDE_AB];

    uint32_t RC[M_TILES][N_TILES][2] = {0};

    int k_tiles = K / WARP_K;
    for (int k_tile = 0; k_tile < k_tiles; k_tile += CHUNK) {
        // A : global to share
        for (int i = 0; i < BLOCK_ROWS / BLOCK_WARPS / 8; ++i) {
            int shared_row = wrap_id * BLOCK_ROWS / BLOCK_WARPS + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_row = block_tile_i * BLOCK_ROWS + shared_row;
            int block_col = k_tile * CHUNK * WARP_K + shared_col;
            *((int4 *)(smem[shared_row] + shared_col)) = *((int4 *)(A + M * block_row + block_col));
        }
        // B : global to share
        for (int i = 0; i < BLOCK_COLS / BLOCK_WARPS / 8; ++i) {
            int shared_row = wrap_id * BLOCK_COLS / BLOCK_WARPS + lane_id / 4 + i * 8;
            int shared_col = lane_id % 4 * 8;
            int block_col = block_tile_j * BLOCK_COLS + shared_row;
            int block_row = k_tile * CHUNK * WARP_K + shared_col;
            *((int4 *)(smem[BLOCK_ROWS + shared_row] + shared_col)) = *((int4 *)(B + N * block_col + block_row));
        }
        __syncthreads();

        for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; warp_tile_m++) {
            for (int warp_tile_n = 0; warp_tile_n < WARP_TILES_N; warp_tile_n++) {
                uint32_t RA[4];
                uint32_t RB[2];
            }
        }
    }
}

size_t initMmaBaseTest() {
    size_t ab_shared_size = (BLOCK_ROWS + BLOCK_COLS) * SHARED_STRIDE_AB * sizeof(half);
    size_t c_shared_size = BLOCK_ROWS * SHARED_STRIDE_AB * sizeof(half);
    return max(ab_shared_size, c_shared_size);
}

void mmaBaseTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaBaseTest();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(N, BLOCK_COLS), div_ceil(M, BLOCK_ROWS));

    mmaBaseKernelTest<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}