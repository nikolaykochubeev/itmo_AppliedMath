import numpy as np
from conditional_number import get_conditional_number
from print_Matrix import print_matrix

def generate_gilbert_matrix(k):
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            matrix[i][j] = 1 / (i + j + 1)
    return matrix

def solve_system(func, k):
    A = generate_gilbert_matrix(k)
    print_matrix(A)
    x = np.array([i for i in range(1, k + 1)])
    F = np.dot(A, x)
    x_new = func(A, F)
    F_new = np.dot(A, x)
    error = np.linalg.norm(F - F_new)
    conditional_number = get_conditional_number(A)
    return x_new, error, conditional_number


def solve_systems(func, k):
    error_array = []
    x_new_array = []
    conditional_number_array = []
    for i in range(2, k + 1):
        x_new, error, conditional_number = solve_system(func, i)
        x_new_array.append(x_new)
        error_array.append(error)
        conditional_number_array.append(conditional_number)
        print("k =", i, "\terror =", round(error, 5), "\tconditional number =", conditional_number, "\nx' =", x_new,)
    return error_array, x_new_array, conditional_number_array
