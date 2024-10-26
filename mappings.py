import math


def analyticalDramReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = r1#math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = c2#math.ceil(c2/pu_width)

    #print("DRAM Map: ",dram_map," GB MAP: ",gb_map," R1 Map: ",r1_map, " C1 Map: ",c1_map," R2 Map: ",r2_map," C2 Map: ",c2_map )
    activates = math.ceil(c1_map/dram_map)*math.ceil(r1_map/banks)
    gb_latencies = math.ceil(r1_map/banks)*math.ceil(r2_map/(min(gb_map,dram_map)))*c2_map
    gb_writes = math.ceil(r1/banks)*r2*c2
    dram_writes = math.ceil(r1/banks)*c1 
    dram_latencies = math.ceil(c1_map/dram_map)*math.ceil(r1_map/banks)
    compute_pus = math.ceil(r1_map/banks)*c1_map*c2_map
    read_dram = math.ceil(r1_map/banks)*c2_map*math.ceil(r2_map/(min(gb_map,dram_map)))
    #read_dram = math.ceil(r1_map/banks)*math.ceil(r2_map/(min(gb_map,dram_map)))

    #print("# Activates:", activates, "#GB Latencies: ", gb_latencies, "#Gb Writes: ", gb_writes, "#DRAM Writes: ", dram_writes, "#DRAM Latencies: ", dram_latencies, "#Compute PUs: ", compute_pus, "#DRAM Reads: ",read_dram)


    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    return total_cycles



def analyticalDramReuseCol(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))
    
    r1_map = r1#math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = c2#math.ceil(c2/pu_width)

    #print("DRAM Map: ",dram_map," GB MAP: ",gb_map," R1 Map: ",r1_map, " C1 Map: ",c1_map," R2 Map: ",r2_map," C2 Map: ",c2_map )
    
    activates = math.ceil(r2_map/dram_map)*math.ceil(c2_map/banks)
    gb_latencies = math.ceil(c2_map/banks)*math.ceil(c1_map/(min(gb_map,dram_map)))*r1_map
    gb_writes = math.ceil(c2/banks)*r1*c1
    dram_writes = math.ceil(c2/banks)*r2
    dram_latencies = math.ceil(r2_map/dram_map)*math.ceil(c2_map/banks)
    compute_pus = math.ceil(c2_map/banks)*r1_map*c1_map
    read_dram = math.ceil(c2_map/banks)*math.ceil(c1_map/(min(gb_map,dram_map)))*r1_map

    #print("# Activates:", activates, "#GB Latencies: ", gb_latencies, "#Gb Writes: ", gb_writes, "#DRAM Writes: ", dram_writes, "#DRAM Latencies: ", dram_latencies, "#Compute PUs: ", compute_pus, "#DRAM Reads: ",read_dram)

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    return total_cycles



def analyticalGBReuseCol(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = r1#math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = c2#math.ceil(c2/pu_width)

    
    #print("DRAM Map: ",dram_map," GB MAP: ",gb_map," R1 Map: ",r1_map, " C1 Map: ",c1_map," R2 Map: ",r2_map," C2 Map: ",c2_map )

    activates = math.ceil(c1_map/(min(gb_map,dram_map)))*math.ceil(r1_map/banks)*c2_map
    gb_latencies = math.ceil(r2_map/gb_map)*c2_map
    gb_writes = r2*c2
    dram_writes = math.ceil(r1/banks)*c1
    dram_latencies = math.ceil(c1_map/dram_map)*math.ceil(r1_map/banks)
    compute_pus = math.ceil(r1_map/banks)*c2_map*c1_map
    read_dram = math.ceil(c1_map/(min(gb_map,dram_map)))*math.ceil(r1_map/banks)*c2_map

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    #print("# Activates:", activates, "#GB Latencies: ", gb_latencies, "#Gb Writes: ", gb_writes, "#DRAM Writes: ", dram_writes, "#DRAM Latencies: ", dram_latencies, "#Compute PUs: ", compute_pus, "#DRAM Reads: ",read_dram)

    return total_cycles

def analyticalGBReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = r1#math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = c2#math.ceil(c2/pu_width)

    #print("DRAM Map: ",dram_map," GB MAP: ",gb_map," R1 Map: ",r1_map, " C1 Map: ",c1_map," R2 Map: ",r2_map," C2 Map: ",c2_map )


    activates = math.ceil(r2_map/(min(gb_map,dram_map)))*math.ceil(c2_map/banks)*r1_map
    gb_latencies = math.ceil(c1_map/gb_map)*r1_map
    gb_writes = r1*c1
    dram_writes = math.ceil(c2/banks)*r2
    dram_latencies = math.ceil(r2_map/dram_map)*math.ceil(c2_map/banks)
    compute_pus = math.ceil(c2_map/banks)*r1_map*c1_map
    read_dram = math.ceil(r2_map/(min(gb_map,dram_map)))*math.ceil(c2_map/banks)*r1_map

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    #print("# Activates:", activates, "#GB Latencies: ", gb_latencies, "#Gb Writes: ", gb_writes, "#DRAM Writes: ", dram_writes, "#DRAM Latencies: ", dram_latencies, "#Compute PUs: ", compute_pus, "#DRAM Reads: ",read_dram)


    return total_cycles



def loopDramReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))
    
    activate_count = 0 
    dram_write_count = 0
    dram_write_latency_count = 0
    dram_read_count = 0

    gb_write_latency_count = 0
    gb_write_count = 0 

    compute_pu_count = 0

    r1_map = r1#math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = c2#math.ceil(c2/pu_width)
    commandsLs = []



    assert c1 == r2, "Incompatabile dimensions for matrix multiplication"

    m1 = makeMatrix(r1,c1)

    m2 = makeMatrix(r2,c2)

    m2_trans = transpose(m2)

    min_tiling = min(dram_map,gb_map)

    m1_dram_write_tiled = partition_each_row_grouped(m1,dram_map)

    m1_tiled = partition_each_row_grouped(m1,min_tiling)
    m2_tiled = partition_each_row_grouped(m2_trans,min_tiling)


    assert len(m1_tiled) == len(m1_dram_write_tiled), "# of Rows of different tilings of same Matrix are not the same"

    for m1_row in range(len(m1_tiled)):
        print(m1_row)
        for m1_tile in range(len(m1_tiled[m1_row])):
            print(m1_tile)

def makeMatrix(x,y):
    return [[0 for col in range(y)] for row in range(x)] #generating matrix of data


test1 = makeMatrix(4,11)
test2 = makeMatrix(11,10)



print(test1)


def transpose(list_of_lists):
    # Use zip and unpacking (*) to transpose the list of lists
    return [list(row) for row in zip(*list_of_lists)]


test2_transpose = transpose(test2)

'''
print(test2_transpose)

for r1 in range(len(test1)):
    print(test1[r1],"ROW of M1")
    for r2 in range (len(test2)):
        print(test2[r2],"Row of M2")
        for values in range(len(test2[r2])):
            print(test2[r2][values], "Value of M2")
'''
# Sample list of lists (matrix)
list_of_lists = [
    [1, 2, 3,10,11,12],
    [4, 5, 6,13,14,15],
    [7, 8, 9,16,17,18]
]

# Transpose the matrix
transposed_matrix = transpose(list_of_lists)
#print(len(list_of_lists[0]),"COLS??")
# Display the result
#print(transposed_matrix)


def partition_each_row_grouped(list_of_lists, tile_width):
    # Create a list to store the grouped row tiles
    grouped_tiles = []
    
    # Iterate through each row and partition it into tiles of width `tile_width`
    for row in list_of_lists:
        # Partition the row into sublists (tiles)
        row_tiles = [row[i:i + tile_width] for i in range(0, len(row), tile_width)]
        # Append the tiles for this row as a grouped sublist
        grouped_tiles.append(row_tiles)
    
    return grouped_tiles

# Sample list of lists (matrix) with dimensions 2x3
list_of_lists = [
    [1, 2, 3],
    [4, 5, 6]
]

# Partition each row into tiles of width 2
grouped_tiles = partition_each_row_grouped(test1, 2)
grouped_2 = partition_each_row_grouped(test2_transpose,2)

# Display the result
print(grouped_tiles , "Mat 1")
print(grouped_2,"Mat 2")
print(len(grouped_tiles),"Total Rows")
#assert len(matrix_a[0]) == len(matrix_b), "Row 2 != Col 1. Cant MatMul"


activate_count = 0 
dram_write_count = 0
dram_write_latency_count = 0
dram_read_count = 0

gb_write_latency_count = 0
gb_write_count = 0 

compute_pu_count = 0

for row in range(len(grouped_tiles)):
    #print(rows,"ROW")
    print(grouped_tiles[row],"Row")

    #write values into the DRAM row after activated 


    for tile in range(len(grouped_tiles[row])):
        activate_count+=1 
        dram_write_latency_count+=1        
        print(grouped_tiles[row][tile],"Tile")
        tile_length = len(grouped_tiles[row][tile])

        for row2 in range(len(grouped_2)):
            print(grouped_2[row2]," Row of 2nd Mat")

        for value in range(len(grouped_tiles[row][tile])):
            print(grouped_tiles[row][tile][value],"VALUE")
            if(grouped_tiles[row][tile][value] == 0):
                grouped_tiles[row][tile][value] = 1
                print(grouped_tiles[row][tile][value],"VALUE WRITE")

