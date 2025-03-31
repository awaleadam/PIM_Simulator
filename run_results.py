import mappings
import matplotlib.pyplot as plt 
import numpy as np
import parserser as ps

from collections import Counter

#Newton/SK Hynix params
def newton_software():
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


    m1_r1 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    c1__r2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    m2_c2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]


    #m1_r1 = list(range(5000,10001,100))
    #c1__r2 = list(range(5000,10001,100))
    #m2_c2  = list(range(5000,10001,100))


    #m1_r1 = list(range(1,4001,100))
    #c1__r2 = list(range(1,4001,100))
    #m2_c2  = list(range(1,4001,100))

    total_count=0
    count_a=0
    count_b=0
    count_c=0
    count_d=0

    ls_set = []

    for i in m1_r1:
        for j in c1__r2:
            for k in m2_c2:
                a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                #ls_set.append([b,c])
                ls_set.append([a,b,c,d])
                '''
                if max(a,b,c,d) == a:
                    print("DRAM COL:",i,j,k)
                elif max(a,b,c,d) == b:
                    print("DRAM ROW",i,j,k)
                elif max(a,b,c,d) == c:
                    print("GB COL",i,j,k)
                elif max(a,b,c,d) == d:
                    print("GB ROW",i,j,k)

                if a == b:
                    print(i,j,k,"SAME DRAM")
                if c == d:
                    print(i,j,k,"SAME GB")

                if a and b > 10000000000:
                    print(i,j,k,"VALUES")
                    print(a,b,"DRAM cycles")
                    print(c,d,"GB cycles")
'''             

                if min(a,b,c,d) == a:
                    #print("DRAM COL:",i,j,k)
                    count_a+=1
                elif min(a,b,c,d) == b:
                    #print("DRAM ROW",i,j,k)
                    count_b+=1
                elif min(a,b,c,d) == c:
                    print("GB COL",i,j,k)
                    count_c+=1
                elif min(a,b,c,d) == d:
                    print("GB ROW",i,j,k)
                    count_d+=1

                total_count+=1

    
    print("Count of DRAM Col is: ",count_a, "Percentage is: ", count_a/total_count)
    print("Count of DRAM Row is: ",count_b, "Percentage is: ", count_b/total_count)
    print("Count of GB Col is: ",count_c, "Percentage is: ", count_c/total_count)
    print("Count of GB Row is: ",count_d, "Percentage is: ", count_d/total_count)
    print("Total Count is: ",total_count)
    count_index_0 = 0
    count_index_1 = 0
    count_index_2 = 0
    count_index_3 = 0
    # Iterate through each list and find the index of the minimum value
    for lst in ls_set:
        min_index = lst.index(min(lst))
        
        # Increment the corresponding counter
        if min_index == 0:
            count_index_0 += 1
        elif min_index == 1:
            count_index_1 += 1
        elif min_index == 2:
            count_index_2 += 1
        elif min_index == 3:
            count_index_3 += 1

    # Display the counts
    print(f"DRAM Reuse Col Best Mapping Rate: {count_index_0/len(ls_set)}")
    print(f"DRAM Reuse Row Best Mapping Rate: {count_index_1/len(ls_set)}")
    print(f"GB Reuse Col Best Mapping Rate: {count_index_2/len(ls_set)}")
    print(f"GB Reuse Row Best Mapping Rate: {count_index_3/len(ls_set)}")
    print("Total length: ",len(ls_set))



    #a = mappings.loopDramReuseRow(10000,1000,1000,5000,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

    #print(ls_set)
    
    sorted_set = sorted(ls_set, key=lambda x: x[0])

    #print(sorted_set)

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

def newton_CNN():
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


    m1_r1 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    c1__r2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    m2_c2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]


    #m1_r1 = list(range(5000,10001,100))
    #c1__r2 = list(range(5000,10001,100))
    #m2_c2  = list(range(5000,10001,100))


    #m1_r1 = list(range(1,4001,100))
    #c1__r2 = list(range(1,4001,100))
    #m2_c2  = list(range(1,4001,100))

    total_count=0
    count_a=0
    count_b=0
    count_c=0
    count_d=0


    #assume that DRAM Reuse row is the count of related work
    #min count of each layer will be framework appraoch
    count_soa_alexnet = 0
    count_soa_vgg16 = 0
    count_soa_googlenet = 0
    count_soa_resnet50 = 0
    count_soa_densenet121 = 0
    count_frame_alexnet = 0 
    count_frame_vgg16 = 0
    count_frame_googlenet = 0
    count_frame_resnet50 = 0
    count_frame_densenet121 = 0
    with open("CNN_configs.txt", "r") as file:
        for line in file:
            line = line.strip()
            linls = line.split("\t")
            assert linls[3] == linls[4], "Matrix dimensions invalid to multiply"
            i = int(linls[2])
            j = int(linls[3])
            k = int(linls[5])

            if linls[1] == 'AlexNet':
                a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

                count_soa_alexnet += min(a,b) 
                count_frame_alexnet += min(a,b,c,d)

            elif linls[1] == 'VGG 16':
                a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

                count_soa_vgg16 += min(a,b)
                count_frame_vgg16 += min(a,b,c,d)
            elif linls[1] == 'GoogleNet':
                a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                
                count_soa_googlenet += min(a,b)
                count_frame_googlenet += min(a,b,c,d)
            elif linls[1] == 'ResNet50':
                a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
               
                count_soa_resnet50 += min(a,b)
                count_frame_resnet50 += min(a,b,c,d)
            elif linls[1] == 'DenseNet-121':
                a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

                count_soa_densenet121 += b
                count_frame_densenet121 += min(a,b,c,d)

    
    print("Alexnet SOA:",count_soa_alexnet," Framework", count_frame_alexnet, "Percent over: ", ((count_soa_alexnet-count_frame_alexnet)/count_soa_alexnet))
    print("VGG16 SOA:",count_soa_vgg16," Framework", count_frame_vgg16, "Percent over: ", ((count_soa_vgg16-count_frame_vgg16)/count_soa_vgg16))
    print("Googlenet SOA:",count_soa_googlenet," Framework", count_frame_googlenet, "Percent over: ", ((count_soa_googlenet-count_frame_googlenet)/count_soa_googlenet))
    print("ResNet50 SOA:",count_soa_resnet50," Framework", count_frame_resnet50, "Percent over: ", ((count_soa_resnet50-count_frame_resnet50)/count_soa_resnet50))
    print("DenseNet121 SOA:",count_soa_densenet121," Framework", count_frame_densenet121, "Percent over: ", ((count_soa_densenet121-count_frame_densenet121)/count_soa_densenet121))

  
    categories = ['Alexnet', 'VGG16', 'Googlenet', 'ResNet50', 'DenseNet121']
    soa_values = [count_soa_alexnet, count_soa_vgg16, count_soa_googlenet, count_soa_resnet50, count_soa_densenet121]
    framework_values = [count_frame_alexnet, count_frame_vgg16, count_frame_googlenet, count_frame_resnet50, count_frame_densenet121]

    # Bar width and positions
    x = np.arange(len(categories))  # X-axis positions for categories
    width = 0.35  # Width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, soa_values, width, label='SOA', color='b')
    bars2 = ax.bar(x + width/2, framework_values, width, label='Framework', color='orange')

    # Customizations
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of SOA and Framework for Different Architectures')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.legend()



    # Save the plot as a file
    plt.tight_layout()
    plt.savefig('bar_graph_log_scale.png', dpi=300)  # Save the figure in high resolution

    # Display the plot
    plt.show()


def newton_Tranformer():
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


    m1_r1 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    c1__r2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    m2_c2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]


    #m1_r1 = list(range(5000,10001,100))
    #c1__r2 = list(range(5000,10001,100))
    #m2_c2  = list(range(5000,10001,100))


    #m1_r1 = list(range(1,4001,100))
    #c1__r2 = list(range(1,4001,100))
    #m2_c2  = list(range(1,4001,100))

    total_count=0
    count_a=0
    count_b=0
    count_c=0
    count_d=0


    #assume that DRAM Reuse row is the count of related work
    #min count of each layer will be framework appraoch
    count_soa_vit = 0
    count_soa_bert = 0

    count_frame_vit = 0
    count_frame_bert = 0

    with open("vit_configs.txt", "r") as file:
        for line in file:
            line = line.strip()
            linls = line.split(",")
            assert linls[3] == linls[2], "Matrix dimensions invalid to multiply"
            i = int(linls[1])
            j = int(linls[2])
            k = int(linls[4])

        
            a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

            count_soa_vit += b 
            count_frame_vit += min(a,b,c,d)

    with open("bert_configs.txt", "r") as file:
        for line in file:
            line = line.strip()
            linls = line.split(",")
            assert linls[3] == linls[2], "Matrix dimensions invalid to multiply"
            i = int(linls[1])
            j = int(linls[2])
            k = int(linls[4])

        
            a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
            d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

            count_soa_bert += b 
            count_frame_bert += min(a,b,c,d)
    
    print("VIT SOA:",count_soa_vit," Framework", count_frame_vit, "Percent over: ", ((count_soa_vit-count_frame_vit)/count_soa_vit))
    print("Bert SOA:",count_soa_bert," Framework", count_frame_bert, "Percent over: ", ((count_soa_bert-count_frame_bert)/count_soa_bert))
    
  
    categories = ['VIT', 'BERT']
    soa_values = [count_soa_vit, count_soa_bert]
    framework_values = [count_soa_bert, count_frame_bert]

    # Bar width and positions
    x = np.arange(len(categories))  # X-axis positions for categories
    width = 0.35  # Width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, soa_values, width, label='SOA', color='b')
    bars2 = ax.bar(x + width/2, framework_values, width, label='Framework', color='orange')

    # Customizations
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of SOA and Framework for Different Architectures')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.legend()



    # Save the plot as a file
    plt.tight_layout()
    plt.savefig('transformer_compare.png', dpi=300)  # Save the figure in high resolution

    # Display the plot
    plt.show()



def software_timing_sweep():

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

    #iterate through timing params

    t_read_dram = [1, 10, 50]
    t_act = [5, 50, 100]
    t_write_dram = [1, 10, 50]
    t_write_latency= [0, 1, 10, 30]
    t_gb_latency = [0,1, 10, 30]
    t_gb_write = [1, 10, 50]
    t_compute_mac = [1,5,20]

    m1_r1 = [5000]
    c1__r2 = [10000]
    m2_c2 = [1000]
    i = m1_r1[0]
    j = c1__r2[0]
    k = m2_c2[0]

    a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    print(a,b,c,d)
    if min(a,b,c,d) == a:
        print("DRAM COL:",a)
    elif min(a,b,c,d) == b:
        print("DRAM ROW",b)
        print("Comparison to DRAM Col",((a-b)/a)," Comparison to GB Col", ((c-b)/c), "Comparison to GB Row", ((d-b)/d))
    elif min(a,b,c,d) == c:
        print("GB COL",c)
    elif min(a,b,c,d) == d:
        print("GB ROW",d)
    ls_set = []

    for i in m1_r1:
        for j in c1__r2:
            for k in m2_c2:
                for t_r_dram in t_read_dram:
                    for act in t_act:
                        for w_dram in t_write_dram:
                            for w_lat in t_write_latency:
                                for gb_lat in t_gb_latency:
                                    for w_gb in t_gb_write:
                                        for c_mac in t_compute_mac:
                                            a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
                                            b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
                                            c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
                                            d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
            
                                            ls_set.append([a,b,c,d])



    print(len(ls_set),"LENGTH")
    count_index_0 = 0
    count_index_1 = 0
    count_index_2 = 0
    count_index_3 = 0
    # Iterate through each list and find the index of the minimum value
    for lst in ls_set:
        min_index = lst.index(min(lst))
        
        # Increment the corresponding counter
        if min_index == 0:
            count_index_0 += 1
        elif min_index == 1:
            count_index_1 += 1
        elif min_index == 2:
            count_index_2 += 1
        elif min_index == 3:
            count_index_3 += 1

    # Display the counts
    print(f"DRAM Reuse Col Best Mapping Rate: {count_index_0/len(ls_set)}")
    print(f"DRAM Reuse Row Best Mapping Rate: {count_index_1/len(ls_set)}")
    print(f"GB Reuse Col Best Mapping Rate: {count_index_2/len(ls_set)}")
    print(f"GB Reuse Row Best Mapping Rate: {count_index_3/len(ls_set)}")

    sorted_set = sorted(ls_set, key=lambda x: x[0])

    #print(sorted_set)

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
    plt.xlabel('Software+Timing Configuration')
    plt.ylabel('Number of Cycles')
    plt.title('Software Timing Sweep Of Sk Hynix AIM')
    plt.legend()


    plt.savefig('software_sweep.png', dpi=300)  # Save the figure in high resolution
    # Show the plot
    plt.show()

def software_hardware_sweep():

    newton_bit_widths = [8,16,64,128]

    newton_dram_sizes = [2048, 4096, 8192, 16384]

    newton_gb_sizes = [2048, 4096, 8192, 16384]

    newton_bankss = [2,4,8,16,32]

    newton_pu_widths = [2,4,8,16,32]

    newton_bit_width = 16

    newton_dram_size = 8192

    newton_gb_size = 8192

    newton_banks = 16

    newton_pu_width = 16

    t_r_dram = newton_t_read_mac = 15

    act = newton_t_act = 48

    w_dram = newton_t_write = 4

    w_lat = newton_t_write_latency = 11

    gb_lat = newton_t_gb_latency = 2

    w_gb = newton_t_gb_write = 1

    c_mac = newton_t_compute_mac = 2



    #iterate through software params


 
    #iterate through timing params



    m1_r1 = [5000]
    c1__r2 = [10000]
    m2_c2 = [1000]
    i = m1_r1[0]
    j = c1__r2[0]
    k = m2_c2[0]

    a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)
    print(a,b,c,d)
    if min(a,b,c,d) == a:
        print("DRAM COL:",a)
    elif min(a,b,c,d) == b:
        print("DRAM ROW",b)
        print("Comparison to DRAM Col",((a-b)/a)," Comparison to GB Col", ((c-b)/c), "Comparison to GB Row", ((d-b)/d))
    elif min(a,b,c,d) == c:
        print("GB COL",c)
    elif min(a,b,c,d) == d:
        print("GB ROW",d)
    ls_set = []

    for i in m1_r1:
        for j in c1__r2:
            for k in m2_c2:
                for newton_bit_width in newton_bit_widths:
                    for newton_dram_size in newton_dram_sizes:
                        for newton_gb_size in newton_gb_sizes:
                            for newton_banks in newton_bankss:
                                for newton_pu_width in newton_pu_widths:

                                    a = mappings.analyticalDramReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
                                    b = mappings.analyticalDramReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
                                    c = mappings.analyticalGBReuseCol(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
                                    d = mappings.analyticalGBReuseRow(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, act, t_r_dram, w_dram, w_lat, w_gb, gb_lat, c_mac)
            
                                    ls_set.append([a,b,c,d])



    print(len(ls_set),"LENGTH")
    count_index_0 = 0
    count_index_1 = 0
    count_index_2 = 0
    count_index_3 = 0
    # Iterate through each list and find the index of the minimum value
    for lst in ls_set:
        min_index = lst.index(min(lst))
        
        # Increment the corresponding counter
        if min_index == 0:
            count_index_0 += 1
        elif min_index == 1:
            count_index_1 += 1
        elif min_index == 2:
            count_index_2 += 1
        elif min_index == 3:
            count_index_3 += 1

    # Display the counts
    print(f"DRAM Reuse Col Best Mapping Rate: {count_index_0/len(ls_set)}")
    print(f"DRAM Reuse Row Best Mapping Rate: {count_index_1/len(ls_set)}")
    print(f"GB Reuse Col Best Mapping Rate: {count_index_2/len(ls_set)}")
    print(f"GB Reuse Row Best Mapping Rate: {count_index_3/len(ls_set)}")

    sorted_set = sorted(ls_set, key=lambda x: x[0])

    #print(sorted_set)

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
    plt.xlabel('Software+Hardware Configuration')
    plt.ylabel('Number of Cycles')
    plt.title('Software Hardware Sweep Of Sk Hynix AIM')
    plt.legend()


    plt.savefig('hardware_sweep.png', dpi=300)  # Save the figure in high resolution
    # Show the plot
    plt.show()

def samsung_validate():
    #arch params samsung:

    n_pus = 8

    bits_pu = 256

    n_channels = 1

    bits_dram_row = 8192

    bits_bu = 2048

    bits_pu_output = 128

    n_banks = 16

    #software params samsung

    m = [64,128,256]
    n = [256,512,1024,2048]

    data_width = 16

    #timing params: Dummy for now

    t_activate = 1
    t_rd_all = 1 
    t_act_buffer = 1
    t_wr_buffer = 1
    t_compute_pu_all = 1
    t_rd_pu_all = 1

    #Mapping parameters

    m_tile = 8 #Number of accumulate registers
    n_tile = 128#width of PU
    # Form of Activate All, Compute PU All, Rd All Bank, Rd PU All, Wr BU, Activate Bu
    #pairs are (64,256), (64,512), (64,1024), (64,2048), (128,256), (128,512), (128,1024), (128,2048), (256,256), (256,512), (256,1024), (256,2048)
    counts = [[4,128,2048,9,272,1,]]

    for m_vals in m:
        for n_vals in n:
            print("For pair ",m_vals,n_vals)
            mappings.analyticalIterRowIntraRow(m_vals, n_vals, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, m_tile, n_tile)

#Newton/SK Hynix params
def samsung_software():
    newton_bit_width = 16

    newton_dram_size = 8192

    newton_gb_size = 8192

    newton_banks = 16

    samsung_banks = 16

    newton_pu_width = 16

    newton_t_read_mac = 10

    newton_t_act = 100

    gb_t_activate = 100

    newton_t_write = 2

    newton_t_write_latency = 5

    newton_t_gb_latency = 2

    newton_t_gb_write = 5

    newton_t_compute_mac = 2

    #iterate through software params


    m1_r1 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    c1__r2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]
    m2_c2 = [1,20,50,500,1000,5000,10000,50000,100000,500000,1000000]


    #m1_r1 = list(range(5000,10001,100))
    #c1__r2 = list(range(5000,10001,100))
    #m2_c2  = list(range(5000,10001,100))


    #m1_r1 = list(range(1,4001,100))
    #c1__r2 = list(range(1,4001,100))
    #m2_c2  = list(range(1,4001,100))

    total_count=0
    count_a=0
    count_b=0
    count_c=0
    count_d=0

    ls_set = []

    for i in m1_r1:
        for j in c1__r2:
            for k in m2_c2:
                a = mappings.analyticalDramReuseColSam(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, samsung_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, gb_t_activate, newton_t_compute_mac)
                b = mappings.analyticalDramReuseRowSam(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, samsung_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, gb_t_activate, newton_t_compute_mac)
                c = mappings.analyticalGBReuseColSam(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, samsung_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, gb_t_activate, newton_t_compute_mac)
                d = mappings.analyticalGBReuseRowSam(i,j,j,k,newton_dram_size, newton_gb_size, newton_banks, samsung_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, gb_t_activate, newton_t_compute_mac)
                #ls_set.append([b,c])
                ls_set.append([a,b,c,d])
                '''
                if max(a,b,c,d) == a:
                    print("DRAM COL:",i,j,k)
                elif max(a,b,c,d) == b:
                    print("DRAM ROW",i,j,k)
                elif max(a,b,c,d) == c:
                    print("GB COL",i,j,k)
                elif max(a,b,c,d) == d:
                    print("GB ROW",i,j,k)

                if a == b:
                    print(i,j,k,"SAME DRAM")
                if c == d:
                    print(i,j,k,"SAME GB")

                if a and b > 10000000000:
                    print(i,j,k,"VALUES")
                    print(a,b,"DRAM cycles")
                    print(c,d,"GB cycles")
'''             

                if min(a,b,c,d) == a:
                    #print("DRAM COL:",i,j,k)
                    count_a+=1
                elif min(a,b,c,d) == b:
                    #print("DRAM ROW",i,j,k)
                    count_b+=1
                elif min(a,b,c,d) == c:
                    print("GB COL",i,j,k)
                    count_c+=1
                elif min(a,b,c,d) == d:
                    print("GB ROW",i,j,k)
                    count_d+=1

                total_count+=1

    
    print("Count of DRAM Col is: ",count_a, "Percentage is: ", count_a/total_count)
    print("Count of DRAM Row is: ",count_b, "Percentage is: ", count_b/total_count)
    print("Count of GB Col is: ",count_c, "Percentage is: ", count_c/total_count)
    print("Count of GB Row is: ",count_d, "Percentage is: ", count_d/total_count)
    print("Total Count is: ",total_count)
    count_index_0 = 0
    count_index_1 = 0
    count_index_2 = 0
    count_index_3 = 0
    # Iterate through each list and find the index of the minimum value
    for lst in ls_set:
        min_index = lst.index(min(lst))
        
        # Increment the corresponding counter
        if min_index == 0:
            count_index_0 += 1
        elif min_index == 1:
            count_index_1 += 1
        elif min_index == 2:
            count_index_2 += 1
        elif min_index == 3:
            count_index_3 += 1

    # Display the counts
    print(f"DRAM Reuse Col Best Mapping Rate: {count_index_0/len(ls_set)}")
    print(f"DRAM Reuse Row Best Mapping Rate: {count_index_1/len(ls_set)}")
    print(f"GB Reuse Col Best Mapping Rate: {count_index_2/len(ls_set)}")
    print(f"GB Reuse Row Best Mapping Rate: {count_index_3/len(ls_set)}")
    print("Total length: ",len(ls_set))



    #a = mappings.loopDramReuseRow(10000,1000,1000,5000,newton_dram_size, newton_gb_size, newton_banks, newton_pu_width, newton_bit_width, newton_t_act, newton_t_read_mac, newton_t_write, newton_t_write_latency, newton_t_gb_write, newton_t_gb_latency, newton_t_compute_mac)

    #print(ls_set)
    
    sorted_set = sorted(ls_set, key=lambda x: x[0])

    #print(sorted_set)

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


def sk_results():


    dorn_values = ps.read_model("cost_model_input/dorn.txt")
    gpt2_values = ps.read_model("cost_model_input/gpt2.txt")
    lstm_values = ps.read_model("cost_model_input/lstm.txt")
    rnn_values = ps.read_model("cost_model_input/rnnt_manually_generated.txt")
    stargan_values = ps.read_model("cost_model_input/stargan_generator.txt")
    vit_values = ps.read_model("cost_model_input/vit.txt")
    resnet_values = ps.read_model("cost_model_input/resnet50.txt")



    dorn_cost_sam = explore_sk(dorn_values)
    dorn_cost_model = explore_model_sk(dorn_values)
    
    gpt2_cost_sam = explore_sk(gpt2_values)
    gpt2_cost_model = explore_model_sk(gpt2_values)  

    lstm_cost_sam = explore_sk(lstm_values)   
    lstm_cost_model = explore_model_sk(lstm_values)

    rnn_cost_sam = explore_sk(rnn_values)
    rnn_cost_model = explore_model_sk(rnn_values)

    stargan_cost_sam = explore_sk(stargan_values)
    stargan_cost_model = explore_model_sk(stargan_values)

    vit_cost_sam = explore_sk(vit_values)
    vit_cost_model = explore_model_sk(vit_values)

    resnet_cost_sam = explore_sk(resnet_values)
    resnet_cost_model = explore_model_sk(resnet_values)

    # Prepare data for plotting
    models = ['DORN', 'GPT2', 'LSTM', 'RNN', 'StarGAN', 'ViT', 'ResNet']
    sam_costs = [dorn_cost_sam, gpt2_cost_sam, lstm_cost_sam, rnn_cost_sam, stargan_cost_sam, vit_cost_sam, resnet_cost_sam]
    model_costs = [dorn_cost_model, gpt2_cost_model, lstm_cost_model, rnn_cost_model, stargan_cost_model, vit_cost_model, resnet_cost_model]
    print("SK COSTS",sam_costs)
    print("MODEL COSTS",model_costs)
    # Plotting
    x = np.arange(len(models))  # X-axis positions for models
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, sam_costs, width, label='Sk Hynix', color='b')
    bars2 = ax.bar(x + width/2, model_costs, width, label='Model', color='orange')

    # Customizations
    ax.set_xlabel('Models')
    ax.set_ylabel('Cost')
    ax.set_title('Comparison of Sk Hynix and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.legend()

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('sk_model_comparison.svg', dpi=300)  # Save the figure in high resolution
    plt.show()


def samsung_results():
    #arch params samsung:

    n_pus = 8

    bits_pu = 256

    n_channels = 64

    bits_dram_row = 8192

    bits_bu = 2048

    bits_pu_output = 128

    n_banks = 16

    #Parsing Models

    dorn_values = ps.read_model("cost_model_input/dorn.txt")
    gpt2_values = ps.read_model("cost_model_input/gpt2.txt")
    lstm_values = ps.read_model("cost_model_input/lstm.txt")
    rnn_values = ps.read_model("cost_model_input/rnnt_manually_generated.txt")
    stargan_values = ps.read_model("cost_model_input/stargan_generator.txt")
    vit_values = ps.read_model("cost_model_input/vit.txt")
    resnet_values = ps.read_model("cost_model_input/resnet50.txt")

    #software params samsung

    m = [64,128,256]
    n = [256,512,1024,2048]

    data_width = 16

    #timing params: Dummy for now

    t_activate = 120
    t_rd_all = 33 
    t_act_buffer = 120
    t_wr_buffer = 16
    t_compute_pu_all = 10
    t_rd_pu_all = 120

    dorn_cost_sam = explore_samsung(dorn_values)
    dorn_cost_model = explore_model_samsung(dorn_values)
    
    gpt2_cost_sam = explore_samsung(gpt2_values)
    gpt2_cost_model = explore_model_samsung(gpt2_values)  

    lstm_cost_sam = explore_samsung(lstm_values)   
    lstm_cost_model = explore_model_samsung(lstm_values)

    rnn_cost_sam = explore_samsung(rnn_values)
    rnn_cost_model = explore_model_samsung(rnn_values)

    stargan_cost_sam = explore_samsung(stargan_values)
    stargan_cost_model = explore_model_samsung(stargan_values)

    vit_cost_sam = explore_samsung(vit_values)
    vit_cost_model = explore_model_samsung(vit_values)

    resnet_cost_sam = explore_samsung(resnet_values)
    resnet_cost_model = explore_model_samsung(resnet_values)

    # Prepare data for plotting
    models = ['DORN', 'GPT2', 'LSTM', 'RNN', 'StarGAN', 'ViT', 'ResNet']
    sam_costs = [dorn_cost_sam, gpt2_cost_sam, lstm_cost_sam, rnn_cost_sam, stargan_cost_sam, vit_cost_sam, resnet_cost_sam]
    model_costs = [dorn_cost_model, gpt2_cost_model, lstm_cost_model, rnn_cost_model, stargan_cost_model, vit_cost_model, resnet_cost_model]
    print("SAMSUNG COSTS",sam_costs)
    print("MODEL COSTS",model_costs)
    # Plotting
    x = np.arange(len(models))  # X-axis positions for models
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, sam_costs, width, label='Samsung', color='b')
    bars2 = ax.bar(x + width/2, model_costs, width, label='Model', color='orange')

    # Customizations
    ax.set_xlabel('Models')
    ax.set_ylabel('Cost')
    ax.set_title('Comparison of Samsung and Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_yscale('log')  # Set y-axis to log scale
    ax.legend()

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('samsung_model_comparison.svg', dpi=300)  # Save the figure in high resolution
    plt.show()

def explore_samsung(params_ls):
    t_activate = 120
    t_rd_all = 33 
    t_act_buffer = 120
    t_wr_buffer = 16
    t_compute_pu_all = 10
    t_rd_pu_all = 120

    #arch params samsung:

    n_pus = 8

    bits_pu = 256

    n_channels = 64

    bits_dram_row = 8192

    bits_bu = 2048

    bits_pu_output = 128

    n_banks = 16

    #Software params

    data_width = 16

    cost = 0
    m_tile = 8 #Number of accumulate registers
    n_tile = 128#width of PU

    for vals in params_ls:
        m = vals[1]
        n = vals[2]
        vector_amount = vals[3]
        matrix_amount = vals[4]
        mtile = min(m_tile,m)
        ntile = min(n_tile,n)
        op = vals[0]
        if mtile == 0:
            print("Mtile is 0----------------------------------------------",params_ls)
            mtile = 1
        if ntile == 0: 
            print("Ntile is 0----------------------------------------------",params_ls)
            ntile = 1
        for i in range(vector_amount):
            for j in range(matrix_amount):
                if op == "Gemv":
                    cost += mappings.analyticalIterRowIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile)
                else:
                    cost += mappings.analyticalNoBroadcast(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile)
    return cost
        
def explore_model_samsung(params_ls):
    t_activate = 120
    t_rd_all = 33 
    t_act_buffer = 120
    t_wr_buffer = 16
    t_compute_pu_all = 10
    t_rd_pu_all = 120

    #arch params samsung:

    n_pus = 8

    bits_pu = 256

    n_channels = 64

    bits_dram_row = 8192

    bits_bu = 2048

    bits_pu_output = 128

    n_banks = 16

    #Software params

    data_width = 16

    cost = 0 

    for param in params_ls:
        ntiles = [x for x in range(1, param[2] + 1, 16)]
        #print(param[2],param[1], "____________________HERE___________________________",n)
        mtiles = param[1]
        vector_amount = param[3]
        matrix_amount = param[4]
        m = param[1]
        n = param[2]
        op = param[0]
        cost_tile = []
        for mtile in range(1, mtiles + 1):
            for ntile in ntiles:
                if op == "Gemv":
                #print("For pair ",mtile,ntile)
                #print(type(mtile),type(ntile))
                    cost1 = mappings.analyticalIterRowIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost2 = mappings.analyticalIterRowIntraCol(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost3 = mappings.analyticalIterColIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost4 = mappings.analyticalIterColIntraCol(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost_tile.append(min(cost1, cost2, cost3, cost4))
                else:
                    temp_cost = mappings.analyticalNoBroadcast(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost_tile.append(temp_cost)
        if cost_tile == []:
            print("Cost tile is empty", param)
        cost += min(cost_tile)
    return cost




def explore_sk(params_ls):
    t_activate = 120
    t_rd_all = 33 
    t_act_buffer = 120
    t_wr_buffer = 16
    t_compute_pu_all = 10
    t_rd_pu_all = 120

    #arch params samsung:

    n_pus = 16

    bits_pu = 256

    n_channels = 64

    bits_dram_row = 16384

    bits_bu = 16384

    bits_pu_output = 16

    n_banks = 16

    #Software params

    data_width = 16

    cost = 0
    m_tile = 1 #Number of accumulate registers
    n_tile = 1024#width of PU

    if mtile == 0:
        print("Mtile is 0----------------------------------------------",params_ls)
        mtile = 1
    if ntile == 0: 
        print("Ntile is 0----------------------------------------------",params_ls)
        ntile = 1

    for vals in params_ls:
        m = vals[1]
        n = vals[2]
        vector_amount = vals[3]
        matrix_amount = vals[4]
        mtile = min(m_tile,m)
        ntile = min(n_tile,n)

        op = vals[0]

        for i in range(vector_amount):
            for j in range(matrix_amount):  
                if op == "Gemv":
                    cost += mappings.analyticalIterRowIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile)
                else:
                    cost += mappings.analyticalNoBroadcast(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile)
    return cost
        
def explore_model_sk(params_ls):
    t_activate = 120
    t_rd_all = 33 
    t_act_buffer = 120
    t_wr_buffer = 16
    t_compute_pu_all = 10
    t_rd_pu_all = 120

    #arch params samsung:

    n_pus = 16

    bits_pu = 256

    n_channels = 64

    bits_dram_row = 16384

    bits_bu = 16384

    bits_pu_output = 16

    n_banks = 16


    #Software params

    data_width = 16

    cost = 0 

    for param in params_ls:
        ntiles = [x for x in range(1, param[2] + 1, 16)]
        #print(param[2],param[1], "____________________HERE___________________________",n)
        mtiles = param[1]
        vector_amount = param[3]
        matrix_amount = param[4]
        m = param[1]
        n = param[2]
        op = param[0]
        cost_tile = []
        for mtile in range(1, mtiles + 1):
            for ntile in ntiles:
                #print("For pair ",mtile,ntile)
                #print(type(mtile),type(ntile))
                if op == "Gemv":
                    cost1 = mappings.analyticalIterRowIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost2 = mappings.analyticalIterRowIntraCol(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost3 = mappings.analyticalIterColIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost4 = mappings.analyticalIterColIntraCol(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost_tile.append(min(cost1, cost2, cost3, cost4))
                else:
                    temp_cost = mappings.analyticalNoBroadcast(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all, mtile, ntile, matrix_amount, vector_amount)
                    cost_tile.append(temp_cost)
        cost += min(cost_tile)
    return cost

        #cost += mappings.analyticalIterRowIntraRow(m, n, data_width, n_pus, bits_pu, bits_pu_output, n_channels, bits_dram_row, bits_bu, n_banks, t_activate, t_rd_all, t_act_buffer, t_wr_buffer, t_compute_pu_all, t_rd_pu_all)
#software_timing_sweep()

#newton_software()

#samsung_software()

#newton_CNN()

#newton_Tranformer()

#software_timing_sweep()

#software_hardware_sweep()

#samsung_validate()

samsung_results()

sk_results()