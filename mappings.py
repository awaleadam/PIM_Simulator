import math


def analyticalDramReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = math.ceil(c2/pu_width)

    print("DRAM Map: ",dram_map," GB MAP: ",gb_map," R1 Map: ",r1_map, " C1 Map: ",c1_map," R2 Map: ",r2_map," C2 Map: ",c2_map )
    activates = math.ceil(c1_map/dram_map)*math.ceil(r1_map/banks)
    gb_latencies = math.ceil(r1_map/banks)*math.ceil(r2_map/(min(gb_map,dram_map)))*c2_map
    gb_writes = math.ceil(r1/banks)*r2*c2
    dram_writes = math.ceil(r1/banks)*c1 
    dram_latencies = math.ceil(c1_map/dram_map)*math.ceil(r1_map/banks)
    compute_pus = math.ceil(r1_map/banks)*c1_map*c2_map
    read_dram = math.ceil(r1_map/banks)*c2_map*math.ceil(r2_map/(min(gb_map,dram_map)))

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    return total_cycles



def analyticalDramReuseCol(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = math.ceil(c2/pu_width)

    
    activates = math.ceil(r2_map/dram_map)*math.ceil(c2_map/banks)
    gb_latencies = math.ceil(c2_map/banks)*math.ceil(c1_map/(min(gb_map,dram_map)))*r1_map
    gb_writes = math.ceil(c2/banks)*r1*c1
    dram_writes = math.ceil(c2/banks)*r2
    dram_latencies = math.ceil(r2_map/dram_map)*math.ceil(c2_map/banks)
    compute_pus = math.ceil(c2_map/banks)*r1_map*c1_map
    read_dram = math.ceil(c2_map/banks)*math.ceil(c1_map/(min(gb_map,dram_map)))*r1_map

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    return total_cycles



def analyticalGBReuseCol(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = math.ceil(c2/pu_width)

    
    print("DRAM Map: ",dram_map," GB MAP: ",gb_map," R1 Map: ",r1_map, " C1 Map: ",c1_map," R2 Map: ",r2_map," C2 Map: ",c2_map )

    activates = math.ceil(c1_map/(min(gb_map,dram_map)))*math.ceil(r1_map/banks)*c2_map
    gb_latencies = math.ceil(r2_map/gb_map)*c2_map
    gb_writes = r2*c2
    dram_writes = math.ceil(r1/banks)*c1
    dram_latencies = math.ceil(c1_map/dram_map)*math.ceil(r1_map/banks)
    compute_pus = math.ceil(r1_map/banks)*c2_map*c1_map
    read_dram = math.ceil(c1_map/(min(gb_map,dram_map)))*math.ceil(r1_map/banks)*c2_map

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    return total_cycles

def analyticalGBReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = math.ceil(c2/pu_width)

    
    activates = math.ceil(r2_map/(min(gb_map,dram_map)))*math.ceil(c2_map/banks)*r1_map
    gb_latencies = math.ceil(c1_map/gb_map)*r1_map
    gb_writes = r1*c1
    dram_writes = math.ceil(c2/banks)*r2
    dram_latencies = math.ceil(r2_map/dram_map)*math.ceil(c2_map/banks)
    compute_pus = math.ceil(c2_map/banks)*r1_map*c1_map
    read_dram = math.ceil(r2_map/(min(gb_map,dram_map)))*math.ceil(c2_map/banks)*r1_map

    total_cycles = activates*activate_time + gb_latencies*gb_write_latency + gb_writes*gb_write_time + dram_writes*dram_write_time + dram_latencies*dram_write_latency + compute_pus*pu_time + read_dram*dram_read_time

    return total_cycles



def loopDramReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = math.ceil(c2/pu_width)
    commandsLs = []




def makeMatrix(x,y):
    return [[0 for col in range(x)] for row in range(y)] #generating matrix of data


test1 = makeMatrix(11,4)
print(test1)

def transpose(list_of_lists):
    # Use zip and unpacking (*) to transpose the list of lists
    return [list(row) for row in zip(*list_of_lists)]

# Sample list of lists (matrix)
list_of_lists = [
    [1, 2, 3,10,11,12],
    [4, 5, 6,13,14,15],
    [7, 8, 9,16,17,18]
]

# Transpose the matrix
transposed_matrix = transpose(list_of_lists)

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
grouped_tiles = partition_each_row_grouped(test1, 8)

# Display the result
print(grouped_tiles , "here")

for rows in grouped_tiles:
    print(rows,"ROW")
    for tiles in rows:
        print(tiles,"Tile")