
MMA_K = 16
PERMUTED_COLS = 4
PERMUTED_OFFSET = 8
AB_SMEM_STRIDE = 32
SMEM_BANK_ROWS = 2


def main():
    for lane_id in range(32):
        p_rol = lane_id % 16
        p_col = (int((lane_id / 16)) * 8 + int((lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS) * PERMUTED_OFFSET) % AB_SMEM_STRIDE
        print("lane id {}, p_rol {}, p_col {}".format(lane_id, p_rol, p_col))



if __name__ == "__main__":
    main()
