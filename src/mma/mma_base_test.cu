#include "common.h"

#define WARP_M 16
#define WARP_N 8
#define WARP_K 16

#define THREADS_PER_BLOCK 256
#define BLOCK_ROWS 256
#define BLOCK_COLS 128

#define M_TILES 16   // BLOCK_ROWS/WARP_M
#define N_TILES 16   // BLOCK_COLS/WARP_N

#define CHUNK 2



__global__ void mmaBaseKernelTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {

}


size_t initMmaBaseTest() {


}

void mmaBaseTest(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    static size_t smem_max_size = initMmaBaseTest();

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS));

    mmaBaseKernelTest<<<grid, block, smem_max_size>>>(A, B, C, M, N, K);
}