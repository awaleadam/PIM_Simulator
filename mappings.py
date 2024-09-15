import math


def analyticalDramReuseRow(r1, c1, r2, c2, dram_size, gb_size, banks, pu_width, bitwidth, activate_time, dram_read_time, dram_write_time, dram_write_latency, gb_write_time, gb_write_latency, pu_time):
    dram_map = math.ceil(dram_size/(bitwidth*pu_width))
    gb_map = math.ceil(gb_size/(bitwidth*pu_width))

    r1_map = math.ceil(r1/pu_width)
    c1_map = math.ceil(c1/pu_width)
    r2_map = math.ceil(r2/pu_width)
    c2_map  = math.ceil(c2/pu_width)

    
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



test = [[1,2,3,4],[3,4,5,6],[4,5,6,7], [5,6,7,8]]


test1 = [[1 for col in range(4)] for row in range(3)]

#print(test1)