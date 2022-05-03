from matrix import Matrix
from copy import deepcopy


def hilbert(n: int) -> Matrix:
    matrix = Matrix(n)
    for i in range(n):
        for j in range(n):
            matrix[i][j] = 1/(i + j + 1)
    return matrix


def diagonal_dominant(n: int, amount: int = 10, step: int = 2):
    matrix = Matrix.random(n)
    for i in range(n):
        matrix[i][i] = sum(matrix[i]) + 1
    matrices = [matrix]
    for i in range(1, amount):
        new_matrix = deepcopy(matrix)
        for j in range(n):
            new_matrix[j][j] = new_matrix[j][j] * step
        matrices.append(new_matrix)
        matrix = new_matrix
    return matrices