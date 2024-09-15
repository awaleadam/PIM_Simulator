import mappings
import matplotlib.pyplot as plt 


#Newton/SK Hynix params

newton_bit_width = 16

newton_dram_size = 8192

newton_gb_size = 8192

newton_banks = 16

newton_pu_width = 16

newton_t_read_mac = 15

newton_t_act = 48

newton_t_write = 4

newton_t_write_latency = 11

newton_t_gb_latency = 2

newton_t_gb_write = 1

newton_t_compute_mac = 2

#iterate through software params


m1_r1 = [1,20,50,1000,5000,10000,50000,100000,500000]
c1__r2 = [1,20,50,1000,5000,10000,50000,100000,500000]
m2_c2 = [1,20,50,1000,5000,10000,50000,100000,500000]

ls_set = []

for i in m1_r1:
    for j in c1__r2:
        for k in m2_c2:
            a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            
            ls_set.append([a,b,c,d])

#print(ls_set)
sorted_set = sorted(ls_set, key=lambda x: x[0])

#print(sorted_set)


# List of sets with the original order preserved (as lists)
list_of_sets = [[5, 3, 1], [8, 2, 7], [4, 9, 6]]

# Sort the list of sets based on the first element in each set
sorted_list = sorted(list_of_sets, key=lambda x: x[0])

# Display the sorted list
print("Sorted list of sets:", sorted_list)



transposed_list = list(zip(*sorted_set))

# Plot each series of values
for idx, series in enumerate(transposed_list):
    if idx == 0:
        plt.plot(range(len(series)), series, marker='o', markersize=3,label=f"DRAM Reuse Col")
    elif idx ==1:
        plt.plot(range(len(series)), series, marker='o', markersize=1,label=f"DRAM Reuse Row")
    elif idx == 2:
        plt.plot(range(len(series)), series, marker='o', markersize=1,label=f"GBuf Reuse Col")
    elif idx ==3:
        plt.plot(range(len(series)), series, marker='o', markersize=1,label=f"GBuf Reuse Row")
    
    print("HI",idx)


plt.yscale('log')

# Adding labels and title
plt.xlabel('Software Configuration')
plt.ylabel('Number of Cycles')
plt.title('Software Sweep Of Sk Hynix AIM')
plt.legend()

# Show the plot
plt.show()
