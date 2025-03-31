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

# Example usage
# parse_file('input.csv')

#parse_file("cost_model_input/dorn.txt")