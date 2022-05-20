import numpy as np
import random
from conditional_number import get_conditional_number


def generate_diagonal_matrix(k):
    values = [0, -1, -2, -3, -4, -5, -6]
    noise = 10 ** (-k)
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            matrix[i][j] = random.choice(values)
    for i in range(k):
        matrix[i][i] = -(sum(matrix[i]) - matrix[i][i]) + noise
    return matrix


def solve_d_system(func, k):
    A = generate_diagonal_matrix(k)
    x = np.array([i for i in range(1, k + 1)])
    F = np.dot(A, x)
    x_new = func(A, F)
    F_new = np.dot(A, x_new)
    error = np.linalg.norm(F - F_new)
    conditional_number = get_conditional_number(A)
    return x_new, error, conditional_number

def solve_d_systems(func, k):
    error_array = []
    x_new_array = []
    for i in range(2, k + 1):
        x_new, error, conditional_number = solve_d_system(func, i)
        x_new_array.append(x_new)
        error_array.append(error)
        print("k =", i, "\terror =", round(error, 5), "\tconditional number =", conditional_number, "\nx' =", x_new)