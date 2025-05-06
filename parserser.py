import os
import math
def read_model(file_path):
    command_ls = []

    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            parts = line.split(',')  # Split the line by commas
            vector_amount = 1
            matrix_amount = 1
            operator = parts[0]
            m = int(parts[1])
            n = int(parts[2])

            if len(parts) == 5:
            
                reuse_type_1 = parts[3]
                reuse_amount_1 = int(parts[4])
                if reuse_type_1 == "vector":
                    vector_amount = int(reuse_amount_1)
                elif reuse_type_1 == "matrix":
                    matrix_amount = int(reuse_amount_1)
                #reuse_type_2 = parts[5]
                #reuse_amount_2 = int(parts[6])
            elif len(parts) == 7:

                reuse_type_1 = parts[3]
                reuse_amount_1 = int(parts[4])
                matrix_amount = reuse_amount_1
                reuse_type_2 = parts[5]
                reuse_amount_2 = int(parts[6])
                vector_amount = reuse_amount_2
            
            #print(f"Operator: {operator}, M: {m}, N: {n}, "
            #      f"Reuse Type 1: {reuse_type_1}, Reuse Amount 1: {reuse_amount_1}, "
            #      f"Reuse Type 2: {reuse_type_2}, Reuse Amount 2: {reuse_amount_2}")
            #print("MAtrix amount: ", matrix_amount)
            #print("Vector amount: ", vector_amount)
            command_ls.append([operator, m, n, vector_amount, matrix_amount])
    #print("Command list: ", command_ls)
    return command_ls
def read_HBM_vals():

    directory_path = "power_vals/HBM/"
    out_files = [f for f in os.listdir(directory_path) if f.endswith('.out')]

    hbm_area_dict = {}
    hbm_energy_dict = {}

    for out_file in out_files:
        if not out_file.endswith('.cfg.out'):  # Exclude .cfg.out files
            file_path = os.path.join(directory_path, out_file)
            #print(f"Reading .out file: {file_path}")
            number = out_file.split('.')[0][-1]  # Extract the number before the first dot
            #print(f"Extracted number from .out file: {number}")
            with open(file_path, 'r') as file:
                for line in file:
                    line=line.strip()
                    #print(line)
                    if line.startswith("Number of banks:"):
                        banks = int(line.strip().split()[-1])  # Convert to bits
                        #(f"Banks: {banks}")
                    if line.startswith("Page size (bits):"):
                        row_size = float(line.strip().split()[-1])
                        #print(f"Row Size: {row_size}")
                    if line.startswith("Activation energy:"):
                        actvation_energy = float(line.strip().split()[-2])
                        #print(f"Activation energy: {actvation_energy}")
                    if line.startswith("Read energy:"):
                        read_energy = float(line.strip().split()[-2])
                        #print(f"Read energy: {read_energy}")
                        #print(f"Cache energy write: {write_energy_cache} nj" )
                    if line.startswith("Write energy:"):
                        write_energy = float(line.strip().split()[-2])
                        #print(f"Write energy: {write_energy}")
                    if line.startswith("Precharge energy:"):
                        precharge_energy = float(line.strip().split()[-2])
                        #print(f"Precharge energy: {precharge_energy}")
                    if line.startswith("DRAM core area:"):
                        core_area = float(line.strip().split()[-2])
                        #print(f"DRAM core area: {core_area}")
                    if line.startswith("DRAM area per die:"):
                        die_area = float(line.strip().split()[-2])
                        #print(f"DRAM area per die: {die_area}")
            hbm_area_dict[(banks, row_size)] = core_area
            hbm_energy_dict[(banks, row_size)] = [actvation_energy, read_energy, write_energy, precharge_energy]
                #print(f"Cache area: {area_cache} mm2")
                #print(f"Cache read energy: {read_energy_cache} nj")
                #print(f"Cache write energy: {write_energy_cache} nj")
                #print(f"Cache leakage power: {leakage_power_cache} mw")
    return hbm_area_dict, hbm_energy_dict
def read_DDR_vals():

    directory_path = "power_vals/DDR/"
    out_files = [f for f in os.listdir(directory_path) if f.endswith('.out')]

    ddr_area_dict = {}
    ddr_energy_dict = {}

    count = 0
    for out_file in out_files:
        if not out_file.endswith('.cfg.out'):  # Exclude .cfg.out files
            file_path = os.path.join(directory_path, out_file)
            #print(f"Reading .out file: {file_path}")
            number = out_file.split('.')[0][-1]  # Extract the number before the first dot
            #print(f"Extracted number from .out file: {number}")
            with open(file_path, 'r') as file:
                count += 1
                for line in file:
                    line=line.strip()
                    if line.startswith("Number of banks:"):
                        banks = int(line.strip().split()[-1])  # Convert to bits
                        #print(f"Banks: {banks}")
                    if line.startswith("Page size (bits):"):
                        row_size = float(line.strip().split()[-1])
                        #print(f"Row Size: {row_size}")
                    if line.startswith("Activation energy:"):
                        actvation_energy = float(line.strip().split()[-2])
                        #print(f"Activation energy: {actvation_energy}")
                    if line.startswith("Read energy:"):
                        read_energy = float(line.strip().split()[-2])
                        #print(f"Read energy: {read_energy}")
                        #print(f"Cache energy write: {write_energy_cache} nj" )
                    if line.startswith("Write energy:"):
                        write_energy = float(line.strip().split()[-2])
                        #print(f"Write energy: {write_energy}")
                    if line.startswith("Precharge energy:"):
                        precharge_energy = float(line.strip().split()[-2])
                        #print(f"Precharge energy: {precharge_energy}")
                    if line.startswith("DRAM core area:"):
                        core_area = float(line.strip().split()[-2])
                        #print(f"DRAM core area: {core_area}")
                    if line.startswith("DRAM area per die:"):
                        die_area = float(line.strip().split()[-2])
                        #print(f"DRAM area per die: {die_area}")
            ddr_area_dict[(banks, row_size)] = core_area
            ddr_energy_dict[(banks, row_size)] = [actvation_energy, read_energy, write_energy, precharge_energy]
                #print(f"Cache area: {area_cache} mm2")
                #print(f"Cache read energy: {read_energy_cache} nj")
                #print(f"Cache write energy: {write_energy_cache} nj")
                #print(f"Cache leakage power: {leakage_power_cache} mw")
    #print(count)
    return ddr_area_dict, ddr_energy_dict


def arch_explore_samsung():
    directory_path = "Arch_explore/Samsung/"
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    channel = 1
    BU = 1
    DRAM = 1
    PU = 1
    Bank = 1
    PU_input = 1
    PU_output = 1
    model_latency_ls = []
    model_area_ls = []
    model_energy_ls = []
    samsung_latency_ls = []
    samsung_area_ls = []
    samsung_energy_ls = []
    pu_area_adder = 0.0
    pu_energy_adder = 0.0
    pu_area_mul = 0.0
    pu_energy_mul = 0.0
    pu_area_dflip = 0.0
    pu_energy_dflip = 0.0
    hbm_area_dict, hbm_energy_dict = read_HBM_vals()

    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        #print(f"Reading file: {file_path}")
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if line.startswith("Arch_Param:"):
                    temp_list = (line.split(":")[1].replace("[", "").replace("]", "")).split(",")  # Extract the value after "Arch Param:" and remove all "[" and "]"
                    #print(temp_list)  # Split the line by commas and print the result  
                    channel = int(temp_list[0])
                    BU = int(temp_list[1])
                    DRAM = int(temp_list[2])
                    PU = int(temp_list[3]) 
                    Bank = int(temp_list[3])
                    PU_input = int(temp_list[4])
                    PU_output = int(temp_list[5])
                    #(f"Channel: {channel}, BU: {BU}, DRAM: {DRAM}, PU: {PU}, Bank: {Bank}, PU_input: {PU_input}, PU_output: {PU_output}")
                if line.startswith("Samsung_cost:"):
                    if line.startswith("Samsung_cost:"):
                        values = line.split(":")[1].strip().replace("[", "").replace("]", "").replace(",", "").split(" ")
                        activate_bank = list(map(float, values[1::7]))  # Convert every 7th element starting from the first index to float
                        cycles_models = list(map(float, values[::7]))  # Convert every 7th element starting from the zeroth index to float
                        compute_pu = list(map(float, values[2::7]))  # Convert every 7th element starting from the second index to float
                        rd_bank = list(map(float, values[3::7]))  # Convert every 7th element starting from the third index to float
                        rd_pu = list(map(float, values[4::7]))  # Convert every 7th element starting from the fourth index to float
                        wr_bu = list(map(float, values[5::7]))  # Convert every 7th element starting from the fifth index to float
                        activate_bu = list(map(float, values[6::7]))  # Convert every 7th element starting from the sixth index to float
                        #cycles_models = float(values[::7])  # Get every 8th element
                        #compute_pu = float(values[2::7])  # Get every 8th element starting from the second index
                        #rd_bank = float(values[3::7])  # Get every 8th element starting from the third index
                        #rd_pu = float(values[4::7])  # Get every 8th element starting from the fourth index
                        #wr_bu = float(values[5::7])  # Get every 8th element starting from the fifth index
                        #activate_bu = float(values[6::7])  # Get every 8th element starting from the sixth index
                        #print("-----------------------START-----------------------")
                        #print(f"Activate bank: {activate_bank}")
                        #print(f"Cycles models: {cycles_models}")
                        #print(f"Compute pu: {compute_pu}")
                        #print(f"Read bank: {rd_bank}")
                        #print(f"Read pu: {rd_pu}")
                        #print(f"Write bu: {wr_bu}")
                        #print(f"Activate bu: {activate_bu}")
                        
                        average_cycles = float(sum(cycles_models)) / len(cycles_models) if cycles_models else 0.0
                        #print(f"Average of cycles_models: {average_cycles}")
                        samsung_latency_ls.append(average_cycles)
                        area = float(sum(compute_pu)) 
                        num_pu_mul = math.ceil(PU_input/16)
                        size_flip_flop = math.ceil(PU_output/16)
                        num_pu_add = num_pu_mul - 1
def read_cache_vals():

    directory_path = "power_vals/Cache/"
    cfg_files = [f for f in os.listdir(directory_path) if f.endswith('.cfg')]
    out_files = [f for f in os.listdir(directory_path) if f.endswith('.out')]
    cfg_out_files = [f for f in os.listdir(directory_path) if f.endswith('.cfg.out')]
    '''
    for cfg_file in cfg_files:
        file_path = os.path.join(directory_path, cfg_file)
        #print(f"Reading file: {file_path}")
        number = cfg_file.split('.')[0][-1]  # Extract the number before the first dot
        #print(f"Extracted number from .cfg file: {number}")
        # Process each .cfg file as needed
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("-size (bytes)"):
                    size_cache = int(line.strip().split()[-1])*8  # Convert to bits
                    print(size_cache)
                    #print(f"Cache size in bits: {size_cache}")
                    #print(line.strip().split()[-1])  # Print each line in the .cfg file

    for cfg_out_file in cfg_out_files:
        file_path = os.path.join(directory_path, cfg_out_file)
        #print(f"Reading .cfg.out file: {file_path}")
        number = cfg_out_file.split('.')[0][-1]  # Extract the number before the first dot
        #print(f"Extracted number from .cfg.out file: {number}")
        # Process each .cfg.out file as needed
        with open(file_path, 'r') as file:
            content = file.read()
            #print(f"Content of {cfg_file}:\n{content}")
    '''
    cache_area_dict = {}
    cache_energy_dict = {}
    for out_file in out_files:
        if not out_file.endswith('.cfg.out'):  # Exclude .cfg.out files
            file_path = os.path.join(directory_path, out_file)
            #print(f"Reading .out file: {file_path}")
            number = out_file.split('.')[0][-1]  # Extract the number before the first dot
            #print(f"Extracted number from .out file: {number}")
            with open(file_path, 'r') as file:
                for line in file:
                    line=line.strip()
                    if line.startswith("Total cache size (bytes):"):
                        cache_size = int(line.strip().split()[-1])*8  # Convert to bits
                        print(f"Cache size in bits: {cache_size}")
                    if line.startswith("Fully associative cache array: Area (mm2):"):
                        area_cache = float(line.strip().split()[-1])
                    if line.startswith("Total dynamic read energy per access (nJ):"):
                        read_energy_cache = float(line.strip().split()[-1])
                        #print(f"Cache energy read: {read_energy_cache} nj" )
                    if line.startswith("Total dynamic write energy per access (nJ):"):
                        write_energy_cache = float(line.strip().split()[-1])
                        #print(f"Cache energy write: {write_energy_cache} nj" )
                    if line.startswith("Total leakage power of a bank (mW):"):
                        leakage_power_cache = float(line.strip().split()[-1])
                        #print(f"Cache leakage power: {leakage_power_cache} mw" )
            cache_area_dict[cache_size] = area_cache
            cache_energy_dict[cache_size] = [read_energy_cache, write_energy_cache]
                #print(f"Cache area: {area_cache} mm2")
                #print(f"Cache read energy: {read_energy_cache} nj")
                #print(f"Cache write energy: {write_energy_cache} nj")
                #print(f"Cache leakage power: {leakage_power_cache} mw")
    return cache_area_dict, cache_energy_dict
# Example usage
# parse_file('input.csv')

#parse_file("cost_model_input/dorn.txt")

def test():
    def find_lowest_number_list(pairs,pairs2):
        # `pairs` is a list of tuples where each tuple is (number, corresponding_list)
        return min((pairs, pairs2), key=lambda x: x[0])

    # Example usage
    pairs = [10,[1, 2, 3,4,5,6]
    ]
    pairs2 = [2,[4,5,6,7,8,9]
    ]
    lowest_list3 = [0,[0,0,0,0,1,2]]
    lowest_list = find_lowest_number_list(pairs,pairs2)
    lowest_list3[0] = lowest_list[0] + lowest_list3[0]
    lowest_list3[1] = [lowest_list3[1][0] + lowest_list[1][0],  lowest_list3[1][1] + lowest_list[1][1],  lowest_list3[1][2] + lowest_list[1][2], + lowest_list3[1][3] + lowest_list[1][3],  lowest_list3[1][4] + lowest_list[1][4], lowest_list3[1][5] + lowest_list[1][5]]
    #lowest_list2 = find_lowest_number_list(pairs2)
    #lowest_list3[0] = lowest_list[0] + lowest_list2[0]
    #lowest_list3[1] = [x + y for x, y in zip(lowest_list[1], lowest_list2[1])]
    print(f"The list with the lowest corresponding number is: {lowest_list3}")

#read_cache_vals()

#test()

#x, y = read_HBM_vals()
#print("START")
#print(x)
#print(y)
arch_explore_samsung()