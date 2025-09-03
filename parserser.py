import os
import matplotlib.pyplot as plt
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
    row_set = set()
    flag1,flag2,flag3,flag4,flag5,flag6,flag7,flag8,flag9 = 0,0,0,0,0,0,0,0,0
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
                        banks = int(line.strip().split()[-1])  # Convert to 
                        flag1 = 1
                        #(f"Banks: {banks}")
                    if line.startswith("Page size (bits):"):
                        row_size = float(line.strip().split()[-1])
                        flag2 = 1
                        #print(f"Row Size: {row_size}")
                    if line.startswith("Activation energy:"):
                        actvation_energy = float(line.strip().split()[-2])
                        flag3 = 1
                        #print(f"Activation energy: {actvation_energy}")
                    if line.startswith("Read energy:"):
                        read_energy = float(line.strip().split()[-2])
                        flag4 = 1
                        #print(f"Read energy: {read_energy}")
                        #print(f"Cache energy write: {write_energy_cache} nj" )
                    if line.startswith("Write energy:"):
                        write_energy = float(line.strip().split()[-2])
                        flag5 = 1
                        #print(f"Write energy: {write_energy}")
                    if line.startswith("Precharge energy:"):
                        precharge_energy = float(line.strip().split()[-2])
                        flag6 = 1
                        #print(f"Precharge energy: {precharge_energy}")
                    if line.startswith("DRAM core area:"):
                        core_area = float(line.strip().split()[-2])
                        flag7 = 1
                        #print(f"DRAM core area: {core_area}")
                    if line.startswith("DRAM area per die:"):
                        die_area = float(line.strip().split()[-2])
                        flag8 = 1
                        #print(f"DRAM area per die: {die_area}")
                    if line.startswith("# rows in subarray:"):
                        rows = int(line.strip().split()[-1])
                        flag9 = 1
                        row_set.add(rows)
                        #print(f"Rows: {rows}")  
            if flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7 and flag8 and flag9:
                if (banks, row_size) not in hbm_area_dict:
                    hbm_area_dict[(banks, row_size)] = []
                    hbm_energy_dict[(banks, row_size)] = []
                #print(ddr_area_dict[(banks, row_size)], "VALUE")
                #print(ddr_area_dict)
                #print(core_area, "CORE AREA")
                hbm_area_dict[(banks, row_size)].append(core_area)
                hbm_energy_dict[(banks, row_size)].append([actvation_energy, read_energy, write_energy, precharge_energy])
            flag1,flag2,flag3,flag4,flag5,flag6,flag7,flag8,flag9 = 0,0,0,0,0,0,0,0,0                                              
            #hbm_area_dict[(banks, row_size)] = core_area
            #hbm_energy_dict[(banks, row_size)] = [float(actvation_energy), float(read_energy), float(write_energy), float(precharge_energy)]
                #print(f"Cache area: {area_cache} mm2")
                #print(f"Cache read energy: {read_energy_cache} nj")
                #print(f"Cache write energy: {write_energy_cache} nj")
                #print(f"Cache leakage power: {leakage_power_cache} mw")
    #print("Row set: ", row_set)
    return hbm_area_dict, hbm_energy_dict
def read_DDR_vals():

    directory_path = "power_vals/DDR/"
    out_files = [f for f in os.listdir(directory_path) if f.endswith('.out')]

    ddr_area_dict = {}
    ddr_energy_dict = {}
    row_set = set()
    flag1,flag2,flag3,flag4,flag5,flag6,flag7,flag8,flag9 = 0,0,0,0,0,0,0,0,0
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
                        flag1 = 1
                        #print(f"Banks: {banks}")
                    if line.startswith("Page size (bits):"):
                        row_size = float(line.strip().split()[-1])
                        flag2 = 1
                        #print(f"Row Size: {row_size}")
                    if line.startswith("Activation energy:"):
                        actvation_energy = float(line.strip().split()[-2])
                        flag3 = 1
                        #print(f"Activation energy: {actvation_energy}")
                    if line.startswith("Read energy:"):
                        read_energy = float(line.strip().split()[-2])
                        flag4 = 1
                        #print(f"Read energy: {read_energy}")
                        #print(f"Cache energy write: {write_energy_cache} nj" )
                    if line.startswith("Write energy:"):
                        write_energy = float(line.strip().split()[-2])
                        flag5 = 1
                        #print(f"Write energy: {write_energy}")
                    if line.startswith("Precharge energy:"):
                        precharge_energy = float(line.strip().split()[-2])
                        flag6 = 1
                        #print(f"Precharge energy: {precharge_energy}")
                    if line.startswith("DRAM core area:"):
                        core_area = float(line.strip().split()[-2])
                        flag7 = 1
                        #print(f"DRAM core area: {core_area}")
                    if line.startswith("DRAM area per die:"):
                        die_area = float(line.strip().split()[-2])
                        flag8 = 1
                        #print(f"DRAM area per die: {die_area}")
                    if line.startswith("# rows in subarray:"):
                        rows = int(line.strip().split()[-1])
                        row_set.add(rows)
                        flag9 = 1
                        #print(f"Rows: {rows}")
            if flag1 and flag2 and flag3 and flag4 and flag5 and flag6 and flag7 and flag8 and flag9:
                if (banks, row_size) not in ddr_area_dict:
                    ddr_area_dict[(banks, row_size)] = []
                    ddr_energy_dict[(banks, row_size)] = []
                #print(ddr_area_dict[(banks, row_size)], "VALUE")
                #print(ddr_area_dict)
                #print(core_area, "CORE AREA")
                ddr_area_dict[(banks, row_size)].append(core_area)
                ddr_energy_dict[(banks, row_size)].append([actvation_energy, read_energy, write_energy, precharge_energy])
            flag1,flag2,flag3,flag4,flag5,flag6,flag7,flag8,flag9 = 0,0,0,0,0,0,0,0,0
                #print(f"Cache area: {area_cache} mm2")
                #print(f"Cache read energy: {read_energy_cache} nj")
                #print(f"Cache write energy: {write_energy_cache} nj")
                #print(f"Cache leakage power: {leakage_power_cache} mw")
    #print(count)
    #print("Row set: ", row_set)
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
    model_latency_compare=[]
    model_area_ls = []
    model_energy_ls = []
    samsung_latency_ls = []
    samsung_latency_compare = []
    samsung_area_ls = []
    samsung_energy_ls = []
    pu_area_adder = .006000
    pu_energy_adder = 700000000000000 #2300000000000000
    pu_area_mul = .012000
    pu_energy_mul = 1500000000000000 #5000000000000000
    pu_area_dflip = 0.00003
    pu_energy_dflip = 101000 #336000
    hbm_area_dict, hbm_energy_dict = read_HBM_vals()
    cache_area_dict, cache_energy_dict = read_cache_vals()
    t_activate = 28
    t_rd_all = 20 
    t_act_buffer = 0
    t_wr_buffer = 2
    t_compute_pu_all = 4
    t_rd_pu_all = 22
    bank_ls = [4, 8, 16, 32]

    other_flag = 0
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        #print(f"Reading file: {file_path}")
        if "other" in file_name:
            other_flag = 1
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
                        #average_cycles2 = (float(sum(activate_bank)) / len(activate_bank))* t_activate + (float(sum(rd_bank)) / len(rd_bank)) * t_rd_all + (float(sum(rd_pu)) / len(rd_pu)) * t_rd_pu_all + (float(sum(wr_bu)) / len(wr_bu)) * t_wr_buffer + (float(sum(activate_bu)) / len(activate_bu)) * t_act_buffer + (float(sum(compute_pu)) / len(compute_pu)) * t_compute_pu_all
                        #average_cycles2 = (float(sum(activate_bank)) / len(activate_bank))* t_activate + (float(sum(rd_bank)) / len(rd_bank)) * t_rd_all + (float(sum(rd_pu)) / len(rd_pu)) * t_rd_pu_all + (float(sum(wr_bu)) / len(wr_bu)) * t_wr_buffer + (float(sum(compute_pu)) / len(compute_pu)) * t_compute_pu_all
                        #samsung_latency_ls.append(average_cycles2)
                        #print(f"Average of cycles_models: {average_cycles2}")
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    samsung_latency_compare.append(average_cycles)
                                    #samsung_latency_compare.append(average_cycles2)
                        num_pu_mul = math.ceil(PU_input/16)
                        size_flip_flop = math.ceil(PU_output/16)
                        num_pu_add = num_pu_mul - 1
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + (bank_value * x)* channel 
                                    elif other_flag == 1:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + cache_area_dict[BU]* channel + (bank_value * x)* channel 
                                    samsung_area_ls.append(area)
                                    #samsung_area_ls.append(area2)
                            #area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + (Bank * x)* channel 
                            #area2 = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + cache_area_dict[BU]* channel + (Bank * x)* channel 
                            #samsung_area_ls.append(area)
                            #samsung_area_ls.append(area2)
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_energy_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * x[1] * channel + (float(sum(activate_bu)) / len(activate_bu)) * (x[0] + x[3]) * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    if other_flag == 1:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * cache_energy_dict[BU][1] * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    samsung_energy_ls.append(power)
                                    #samsung_energy_ls.append(power2)
                if line.startswith("Model_cost:"):
                    if line.startswith("Model_cost:"):
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
                        model_latency_ls.append(average_cycles)
                        #average_cycles2 = (float(sum(activate_bank)) / len(activate_bank))* t_activate + (float(sum(rd_bank)) / len(rd_bank)) * t_rd_all + (float(sum(rd_pu)) / len(rd_pu)) * t_rd_pu_all + (float(sum(wr_bu)) / len(wr_bu)) * t_wr_buffer + (float(sum(activate_bu)) / len(activate_bu)) * t_act_buffer + (float(sum(compute_pu)) / len(compute_pu)) * t_compute_pu_all
                        #average_cycles2 = (float(sum(activate_bank)) / len(activate_bank))* t_activate + (float(sum(rd_bank)) / len(rd_bank)) * t_rd_all + (float(sum(rd_pu)) / len(rd_pu)) * t_rd_pu_all + (float(sum(wr_bu)) / len(wr_bu)) * t_wr_buffer + (float(sum(compute_pu)) / len(compute_pu)) * t_compute_pu_all
                        #model_latency_ls.append(average_cycles2)
                        #print(f"Average of cycles_models: {average_cycles2}")
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:  
                                    model_latency_compare.append(average_cycles)
                                    #model_latency_compare.append(average_cycles2)
                        num_pu_mul = math.ceil(PU_input/16)
                        size_flip_flop = math.ceil(PU_output/16)
                        num_pu_add = num_pu_mul - 1
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + (bank_value * x)* channel 
                                    elif other_flag == 1:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + cache_area_dict[BU]* channel + (bank_value * x)* channel 
                                    model_area_ls.append(area)
                                    #model_area_ls.append(area2)
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_energy_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * x[1] * channel + (float(sum(activate_bu)) / len(activate_bu)) * (x[0] + x[3]) * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    if other_flag == 1:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * cache_energy_dict[BU][1] * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    model_energy_ls.append(power)
                                    #model_energy_ls.append(power2)
            other_flag = 0

    # Plot model_latency_ls and samsung_latency_ls
    # Sort samsung_latency_ls and model_latency_ls based on samsung_latency_ls
    sorted_indices = sorted(range(len(samsung_latency_ls)), key=lambda k: samsung_latency_ls[k])
    samsung_latency_ls = [samsung_latency_ls[i] for i in sorted_indices]
    model_latency_ls = [model_latency_ls[i] for i in sorted_indices]

    sorted_indices2 = sorted(range(len(samsung_latency_compare)), key=lambda k: samsung_latency_compare[k])
    model_energy_ls = [model_energy_ls[i] for i in sorted_indices2]
    samsung_energy_ls = [samsung_energy_ls[i] for i in sorted_indices2]
    samsung_latency_compare = [samsung_latency_compare[i] for i in sorted_indices2]
    model_latency_compare = [model_latency_compare[i] for i in sorted_indices2]

    plt.figure(figsize=(10, 6), dpi=100)

    # Plot the first dataset
    plt.scatter(model_energy_ls, model_latency_compare, label='DPC', color='blue', alpha=0.7, edgecolors='k', s=150)

    # Plot the second dataset
    plt.scatter(samsung_energy_ls, samsung_latency_compare, label='Samsung', color='red', alpha=0.7, edgecolors='k', s=150)

    # Add labels, title, and legend
    plt.title('Energy vs Latency across HBM Configurations of DPC vs Samsung', fontsize=40)
    plt.xlabel('Energy (nJ)', fontsize=40)
    plt.ylabel('Latency (Cycles)', fontsize=40)
    plt.xticks(fontsize=30)  # Increase font size of x ticks
    plt.yticks(fontsize=30)  # Increase font size of y ticks
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend(fontsize=30)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('energy_vs_latency_comparison.svg', format='svg')
    plt.show()


    model_perf = [lat / energy for lat, energy in zip(model_latency_compare, model_energy_ls)]
    samsung_perf = [lat / energy for lat, energy in zip(samsung_latency_compare, samsung_energy_ls)]
    percentage_differences = [
        ((-samsung + model) / samsung) * 100 if samsung != 0 else 0
        for samsung, model in zip(samsung_perf, model_perf)
    ]
    # Calculate average, max, and min percentage differences
    avg_difference = sum(percentage_differences) / len(percentage_differences) if percentage_differences else 0
    max_difference = max(percentage_differences, default=0)
    min_difference = min(percentage_differences, default=0)

    print(f"Average Difference Perf: {avg_difference:.2f}%")
    print(f"Max Difference Perf: {max_difference:.2f}%")
    print(f"Min Difference Perf: {min_difference:.2f}%")


    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.plot([lat / energy for lat, energy in zip(model_latency_compare, model_energy_ls)], label='DPIMC', marker='o', markersize=4)

    # Plot the second dataset
    plt.plot([lat / energy for lat, energy in zip(samsung_latency_compare, samsung_energy_ls)], label='Samsung Compiler', marker='x', markersize=4)


    # Add labels, title, and legend
    plt.title('Performance Comparison')
    plt.xlabel('Configuration')
    plt.ylabel('Perfromance')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('performance_comparison_hbm.svg', format='svg')
    plt.show()


    # Calculate the differences in percentage
    percentage_differences = [
        ((samsung - model) / samsung) * 100 if samsung != 0 else 0
        for samsung, model in zip(samsung_latency_ls, model_latency_ls)
    ]
    # Calculate average, max, and min percentage differences
    avg_difference = sum(percentage_differences) / len(percentage_differences) if percentage_differences else 0
    max_difference = max(percentage_differences, default=0)
    min_difference = min(percentage_differences, default=0)

    print(f"Average Difference: {avg_difference:.2f}%")
    print(f"Max Difference: {max_difference:.2f}%")
    print(f"Min Difference: {min_difference:.2f}%")

    # Plot the sorted latency lists
    plt.figure(figsize=(10, 6))
    plt.plot(model_latency_ls, label='DPC', marker='o', markersize=4)
    plt.plot(samsung_latency_ls, label='Samsung Compiler', marker='x', markersize=4)
    plt.title('Average Latency across HBM Configurations of DPC vs Samsung', fontsize=40)
    plt.xlabel('Architecture Configuration', fontsize=40)
    plt.ylabel('Cycles',fontsize=40)
    plt.yscale('log')
    plt.xticks([])  # Remove the labels on the x-axis
    y_ticks = [10**i for i in range(10, 13)]  # Set y-ticks from 10^10 to 10^12
    plt.yticks(ticks=y_ticks, labels=[f"$10^{{{i}}}$" for i in range(10, 13)], fontsize=30)  # Use 10^x format for y-axis labels
    plt.legend(fontsize=30)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig('latency_compare_hbm.svg', format='svg')
    plt.show()





    # Calculate the differences in percentage for power
    percentage_differences_power = [
        ((samsung - model) / samsung) * 100 if samsung != 0 else 0
        for samsung, model in zip(samsung_energy_ls, model_energy_ls)
    ]

    # Calculate average, max, and min percentage differences for power
    avg_difference_power = sum(percentage_differences_power) / len(percentage_differences_power) if percentage_differences_power else 0
    max_difference_power = max(percentage_differences_power, default=0)
    min_difference_power = min(percentage_differences_power, default=0)

    print(f"Average Power Difference: {avg_difference_power:.2f}%")
    print(f"Max Power Difference: {max_difference_power:.2f}%")
    print(f"Min Power Difference: {min_difference_power:.2f}%")

    # Plot model_energy_ls vs model_area_ls and samsung_energy_ls vs samsung_area_ls
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.scatter(model_area_ls, model_energy_ls, label='DPIMC', color='blue', alpha=0.7, edgecolors='k')

    # Plot the second dataset
    plt.scatter(samsung_area_ls, samsung_energy_ls, label='Samsung Compiler', color='red', alpha=0.7, edgecolors='k')

    # Add labels, title, and legend
    plt.title('Average Energy vs Area Comparison')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Energy (nJ)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('energy_vs_area_comparison.svg', format='svg')
    plt.show()
    # Plot model_area_ls vs model_latency_compare and samsung_area_ls vs samsung_latency_compare
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.scatter(model_area_ls, model_latency_compare, label='DPIMC', color='blue', alpha=0.7, edgecolors='k')

    # Plot the second dataset
    plt.scatter(samsung_area_ls, samsung_latency_compare, label='Samsung Compiler', color='red', alpha=0.7, edgecolors='k')

    # Add labels, title, and legend
    plt.title('Area vs Latency Comparison')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Latency (Cycles)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('area_vs_latency_comparison.svg', format='svg')
    plt.show()
    # Plot energy vs latency for both DPIMC and Samsung Compiler



def arch_explore_sk():
    directory_path = "Arch_explore/SK/"
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    channel = 1
    BU = 1
    DRAM = 1
    PU = 1
    Bank = 1
    PU_input = 1
    PU_output = 1
    model_latency_ls = []
    model_latency_compare=[]
    model_area_ls = []
    model_energy_ls = []
    samsung_latency_ls = []
    samsung_latency_compare = []
    samsung_area_ls = []
    samsung_energy_ls = []
    pu_area_adder = .006000
    pu_energy_adder = 2300000000000000
    pu_area_mul = .012000
    pu_energy_mul = 5000000000000000
    pu_area_dflip = 0.00003
    pu_energy_dflip = 336000
    
    hbm_area_dict, hbm_energy_dict = read_DDR_vals()
    cache_area_dict, cache_energy_dict = read_cache_vals()
    bank_ls = [4, 8, 16, 32]
    t_activate = 96
    t_rd_all = 48 
    t_act_buffer = 96
    t_wr_buffer = 44
    t_compute_pu_all = 4
    t_rd_pu_all = 64

    other_flag = 0 
    for file_name in files:
        file_path = os.path.join(directory_path, file_name)
        if "other" in file_name:
            other_flag = 1
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
                if line.startswith("SK_cost:"):
                    if line.startswith("SK_cost:"):
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
                        #average_cycles2 = (float(sum(activate_bank)) / len(activate_bank))* t_activate + (float(sum(rd_bank)) / len(rd_bank)) * t_rd_all + (float(sum(rd_pu)) / len(rd_pu)) * t_rd_pu_all + (float(sum(wr_bu)) / len(wr_bu)) * t_wr_buffer + (float(sum(activate_bu)) / len(activate_bu)) * t_act_buffer + (float(sum(compute_pu)) / len(compute_pu)) * t_compute_pu_all
                        #samsung_latency_ls.append(average_cycles2)
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    samsung_latency_compare.append(average_cycles)
                                    #samsung_latency_compare.append(average_cycles2)
                        #print(f"Average of cycles_models: {average_cycles2}")
                        
                        num_pu_mul = math.ceil(PU_input/16)
                        size_flip_flop = math.ceil(PU_output/16)
                        num_pu_add = num_pu_mul - 1
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + (bank_value * x)* channel 
                                    elif other_flag == 1:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + cache_area_dict[BU]* channel + (bank_value * x)* channel 
                                    samsung_area_ls.append(area)
                                    #samsung_area_ls.append(area2)
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_energy_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * x[1] * channel + (float(sum(activate_bu)) / len(activate_bu)) * (x[0] + x[3]) * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    elif other_flag == 1:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * cache_energy_dict[BU][1] * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    samsung_energy_ls.append(power)
                                    #samsung_energy_ls.append(power2)
                if line.startswith("Model_cost:"):
                    if line.startswith("Model_cost:"):
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
                        model_latency_ls.append(average_cycles)
                        #average_cycles2 = (float(sum(activate_bank)) / len(activate_bank))* t_activate + (float(sum(rd_bank)) / len(rd_bank)) * t_rd_all + (float(sum(rd_pu)) / len(rd_pu)) * t_rd_pu_all + (float(sum(wr_bu)) / len(wr_bu)) * t_wr_buffer + (float(sum(activate_bu)) / len(activate_bu)) * t_act_buffer + (float(sum(compute_pu)) / len(compute_pu)) * t_compute_pu_all
                        #model_latency_ls.append(average_cycles2)
                        #print(f"Average of cycles_models: {average_cycles2}")
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    model_latency_compare.append(average_cycles)
                                    #model_latency_compare.append(average_cycles2)
                        num_pu_mul = math.ceil(PU_input/16)
                        size_flip_flop = math.ceil(PU_output/16)
                        num_pu_add = num_pu_mul - 1
                        for bank_value in bank_ls:
                            if bank_value >= PU:
                                for x in hbm_area_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + (bank_value * x)* channel 
                                    elif other_flag == 1:
                                        area = PU* (num_pu_add * pu_area_adder + num_pu_mul * pu_area_mul + size_flip_flop * pu_area_dflip) + cache_area_dict[BU]* channel + (bank_value * x)* channel 
                                    model_area_ls.append(area)
                                    #model_area_ls.append(area2)
                        for bank_value in bank_ls:
                            if bank_value >= PU:                        
                                for x in hbm_energy_dict[(bank_value,DRAM)]:
                                    if other_flag == 0:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * x[1] * channel + (float(sum(activate_bu)) / len(activate_bu)) * (x[0] + x[3]) * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    if other_flag == 1:
                                        power = (float(sum(activate_bank)) / len(activate_bank)) * (x[0] + x[3]) * PU * channel + (float(sum(rd_bank)) / len(rd_bank)) * x[1] * PU * channel + (float(sum(rd_pu)) / len(rd_pu)) * (x[0] + x[2] ) * PU * channel + (float(sum(wr_bu)) / len(wr_bu)) * cache_energy_dict[BU][1] * channel + (float(sum(compute_pu)) / len(compute_pu)) * (num_pu_add * pu_energy_adder + num_pu_mul * pu_energy_mul + size_flip_flop * pu_energy_dflip) * PU * channel
                                    model_energy_ls.append(power)
                                    #model_energy_ls.append(power2)
        other_flag = 0

    # Plot model_latency_ls and samsung_latency_ls
    # Sort samsung_latency_ls and model_latency_ls based on samsung_latency_ls
    sorted_indices = sorted(range(len(samsung_latency_ls)), key=lambda k: samsung_latency_ls[k])
    samsung_latency_ls = [samsung_latency_ls[i] for i in sorted_indices]
    model_latency_ls = [model_latency_ls[i] for i in sorted_indices]

    sorted_indices2 = sorted(range(len(samsung_latency_compare)), key=lambda k: samsung_latency_compare[k])
    model_energy_ls = [model_energy_ls[i] for i in sorted_indices2]
    samsung_energy_ls = [samsung_energy_ls[i] for i in sorted_indices2]
    samsung_latency_compare = [samsung_latency_compare[i] for i in sorted_indices2]
    model_latency_compare = [model_latency_compare[i] for i in sorted_indices2]
    # Calculate the differences in percentage
    percentage_differences = [
        ((samsung - model) / samsung) * 100 if samsung != 0 else 0
        for samsung, model in zip(samsung_latency_ls, model_latency_ls)
    ]


    plt.figure(figsize=(10, 6), dpi=100)

    # Plot the first dataset
    plt.scatter(model_energy_ls, model_latency_compare, label='DPC', color='blue', alpha=0.7, edgecolors='k', s=150)

    # Plot the second dataset
    plt.scatter(samsung_energy_ls, samsung_latency_compare, label='SK Hynix', color='red', alpha=0.7, edgecolors='k', s=150)

    # Add labels, title, and legend
    plt.title('Energy vs Latency across DDR Configurations of DPC vs SK Hynix', fontsize=40)
    plt.xlabel('Energy (nJ)', fontsize=40)
    plt.ylabel('Latency (Cycles)', fontsize=40)
    plt.xticks(fontsize=30)  # Increase font size of x ticks
    plt.yticks(fontsize=30)  # Increase font size of y ticks
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend(fontsize=30)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('energy_vs_latency_comparison.svg', format='svg')
    plt.show()


    # Print negative differences along with corresponding elements in samsung and model lists
    for diff, samsung, model in zip(percentage_differences, samsung_latency_ls, model_latency_ls):
        if diff < 0:
            print(f"Negative Difference: {diff:.2f}%, Samsung: {samsung}, Model: {model}")

    # Calculate average, max, and min percentage differences
    avg_difference = sum(percentage_differences) / len(percentage_differences) if percentage_differences else 0
    max_difference = max(percentage_differences, default=0)
    min_difference = min(percentage_differences, default=0)

    print(f"Average Difference: {avg_difference:.2f}%")
    print(f"Max Difference: {max_difference:.2f}%")
    print(f"Min Difference: {min_difference:.2f}%")

    # Plot the sorted latency lists
    plt.figure(figsize=(10, 6))
    plt.plot(model_latency_ls, label='DPC', marker='o', markersize=4)
    plt.plot(samsung_latency_ls, label='SK Hynix Compiler', marker='x', markersize=4)
    plt.title('Average Latency across DDR Configurations of DPC vs SK Hynix', fontsize=40)
    plt.xlabel('Architecture Configuration', fontsize=40)
    plt.ylabel('Cycles', fontsize=40)
    #plt.yscale('log')
    plt.xticks([])  # Remove the labels on the x-axis
    y_ticks = [10**i for i in range(10, 13)]  # Set y-ticks from 10^10 to 10^12
    plt.yticks(ticks=y_ticks, labels=[f"$10^{{{i}}}$" for i in range(10, 13)], fontsize=17)  # Use 10^x format for y-axis labels
    plt.legend(fontsize=30)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig('latency_compare_ddr.svg', format='svg')
    plt.show()




    # Calculate the differences in percentage for power
    percentage_differences_power = [
        ((samsung - model) / samsung) * 100 if samsung != 0 else 0
        for samsung, model in zip(samsung_energy_ls, model_energy_ls)
    ]

    # Calculate average, max, and min percentage differences for power
    avg_difference_power = sum(percentage_differences_power) / len(percentage_differences_power) if percentage_differences_power else 0
    max_difference_power = max(percentage_differences_power, default=0)
    min_difference_power = min(percentage_differences_power, default=0)

    print(f"Average Power Difference DDR: {avg_difference_power:.2f}%")
    print(f"Max Power Difference: {max_difference_power:.2f}%")
    print(f"Min Power Difference: {min_difference_power:.2f}%")

    # Plot model_energy_ls vs model_area_ls and samsung_energy_ls vs samsung_area_ls
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.scatter(model_area_ls, model_energy_ls, label='DPIMC', color='blue', alpha=0.7, edgecolors='k')

    # Plot the second dataset
    plt.scatter(samsung_area_ls, samsung_energy_ls, label='SK Hynix Compiler', color='red', alpha=0.7, edgecolors='k')

    # Add labels, title, and legend
    plt.title('Average Energy vs Area Comparison')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Energy (nJ)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('energy_vs_area_comparison_ddr.svg', format='svg')
    plt.show()


    # Plot model_energy_ls vs model_latency_compare and samsung_energy_ls vs samsung_latency_compare
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.scatter(model_energy_ls, model_latency_compare, label='DPIMC', color='blue', alpha=0.7, edgecolors='k')

    # Plot the second dataset
    plt.scatter(samsung_energy_ls, samsung_latency_compare, label='SK Hynix Compiler', color='red', alpha=0.7, edgecolors='k')

    # Add labels, title, and legend
    plt.title('Energy vs Latency Comparison')
    plt.xlabel('Energy (nJ)')
    plt.ylabel('Latency (Cycles)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('energy_vs_latency_comparison_ddr.svg', format='svg')
    plt.show()
    # Plot model_area_ls vs model_latency_compare and samsung_area_ls vs samsung_latency_compare
    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.scatter(model_area_ls, model_latency_compare, label='DPIMC', color='blue', alpha=0.7, edgecolors='k')

    # Plot the second dataset
    plt.scatter(samsung_area_ls, samsung_latency_compare, label='SK Hynix Compiler', color='red', alpha=0.7, edgecolors='k')

    # Add labels, title, and legend
    plt.title('Area vs Latency Comparison')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Latency (Cycles)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)



    model_perf = [lat / energy for lat, energy in zip(model_latency_compare, model_energy_ls)]
    samsung_perf = [lat / energy for lat, energy in zip(samsung_latency_compare, samsung_energy_ls)]
    percentage_differences = [
        ((-samsung + model) / samsung) * 100 if samsung != 0 else 0
        for samsung, model in zip(samsung_perf, model_perf)
    ]

    # Calculate average, max, and min percentage differences
    avg_difference = sum(percentage_differences) / len(percentage_differences) if percentage_differences else 0
    max_difference = max(percentage_differences, default=0)
    min_difference = min(percentage_differences, default=0)

    print(f"Average Difference Perf: {avg_difference:.2f}%")
    print(f"Max Difference Perf: {max_difference:.2f}%")
    print(f"Min Difference Perf: {min_difference:.2f}%")



    plt.figure(figsize=(10, 6))

    # Plot the first dataset
    plt.plot([lat / energy for lat, energy in zip(model_latency_compare, model_energy_ls)], label='DPIMC', marker='o', markersize=4)

    # Plot the second dataset
    plt.plot([lat / energy for lat, energy in zip(samsung_latency_compare, samsung_energy_ls)], label='SK Hynix Compiler', marker='x', markersize=4)

    # Add labels, title, and legend
    plt.title('Performance Comparison')
    plt.xlabel('Configuration')
    plt.ylabel('Perfromance')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save and show the plot
    plt.savefig('performance_comparison_ddr.svg', format='svg')
    plt.show()
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
                        #print(f"Cache size in bits: {cache_size}")
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
#print(y[(32,1024.0)][0]*2.0)

arch_explore_samsung()
#arch_explore_sk()