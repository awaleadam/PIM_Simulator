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
