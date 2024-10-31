import mappings

def traverse_n_rows_at_a_time(matrix, n):
    # Iterate over the matrix in steps of `n`
    count=0
    for i in range(0, len(matrix), n):
        # Get the current chunk of `n` rows
        print("I value",i)
        count+=1
        rows_chunk = matrix[i:i + n]
        print("Length",len(rows_chunk))
        print("Matrix current row: ",matrix[i])
        print(f"Rows {i+1} to {i+len(rows_chunk)}:", rows_chunk)
    print("Count:",count)
# Updated example matrix
matrix = [
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [1, 3, 2],
    [2, 7, 3],
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [1, 3, 2],
    [2, 7, 3],
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [1, 3, 2],
    [2, 7, 3],
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [1, 3, 2],
    [2, 7, 3],
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [1, 3, 2],
    [2, 7, 3],
    [1, 3, 5],
    [2, 4, 6],
    [7, 8, 9],
    [1, 3, 2],
    [2, 7, 3],
    [2, 7, 3],
    [2, 7, 3],
    [2, 7, 3]
]

# Traverse the matrix 2 rows at a time
traverse_n_rows_at_a_time(matrix, 4)


def hardware_software_sweep():

    newton_bit_width = 16

    newton_dram_size = 8192

    newton_gb_size = 8192

    newton_banks = 16

    newton_pu_width = 16

    newton_t_read_mac = 10

    newton_t_act = 100

    newton_t_write = 2

    newton_t_write_latency = 5

    newton_t_gb_latency = 1

    newton_t_gb_write = 1

    newton_t_compute_mac = 2

    #iterate through software params


    m1_r1 = [1,20,50,1000,5000,10000,50000,100000,500000]
    c1__r2 = [1,20,50,1000,5000,10000,50000,100000,500000]
    m2_c2 = [1,20,50,1000,5000,10000,50000,100000,500000]


    #m1_r1 = list(range(5000,10001,100))
    #c1__r2 = list(range(5000,10001,100))
    #m2_c2  = list(range(5000,10001,100))


    #m1_r1 = list(range(1,4001,100))
    #c1__r2 = list(range(1,4001,100))
    #m2_c2  = list(range(1,4001,100))

    m1_r1 = [10000]
    c1__r2 = [1000]
    m2_c2 = [5000]


    #Arch parameters

    banks = [32,16,2]#[32,16,8,4,2]
    pu_widths = [32,16,2]#[32,16,8,4,2]
    bit_widths = [128,32,4]
    dram = [32768,16384,8192,2048]
    gb = [32768,16384,8192,2048]
    total_count=0
    count_a=0
    count_b=0
    count_c=0
    count_d=0

    ls_set = []
    counting =0
    for i in m1_r1:
        for j in c1__r2:
            for k in m2_c2:
                for bank in banks:
                    for pu in pu_widths:
                        for bits in bit_widths:
                            for drams in dram:
                                for gbs in gb:
                                    counting+=1
                                    if counting == 236 or counting == 232 or counting == 228 or counting == 220 or counting == 216 or counting == 215 or counting == 258 or counting == 252 or counting == 248 or counting == 247 or counting == 244 or counting == 243 or counting == 242 or counting == 212 or counting == 211 or counting == 210:
                                        print("BANKS: ", bank, " PU Width: ",pu, "Bit length: ", bits, "DRAM Size", drams, "GB Size", gbs, "For iteration count of: ",counting)
                                        mappings.analyticalDramReuseRow(i,j,j,k,drams, gbs, bank, pu, bits, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                                        print()
hardware_software_sweep()