import numpy as np
import random
import copy
from scipy.sparse import eye
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import math


def get_conditional_number(A):
    A_reverse = np.fliplr(A)
    return np.linalg.norm(A) * np.linalg.norm(A_reverse)


def generate_diagonal_matrix(k):
    values = [0.0, -1.0, -2.0, -3.0, -4.0]
    noise = 10 ** (-k)
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            matrix[i, j] = random.choice(values)

    for j in range(k):
        for i in range(j):
            matrix[j, i] = matrix[i, j]

    for i in range(k):
        matrix[i, i] = -(sum(matrix[i]) - matrix[i, i]) + noise
    return matrix


def generate_gilbert_matrix(k):
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            matrix[i][j] = 1 / (i + j + 1)
    return matrix


def stop_method(A, eps):
    n, m = A.shape
    s = 0
    for j in range(n):
        for i in range(j):
            s += A[i, j] * A[i, j]

    return np.square(1 / 2 * s) < eps


def findMax(A):
    n, m = A.shape
    max = -1
    max_i = -1
    max_j = -1
    for j in range(n):
        for i in range(j):
            if max < abs(A[i, j]):
                max = abs(A[i, j])
                max_i = i
                max_j = j

    return A[max_i, max_j], max_i, max_j


def findU(A):
    n, m = A.shape
    max, max_i, max_j = findMax(A)
    if A[max_i, max_i] == A[max_j, max_j]:
        fi = np.pi / 4.0
    else:
        fi = np.arctan(2.0 * max / (A[max_i, max_i] - A[max_j, max_j])) / 2.0
    U = eye(n, m, format="lil")
    U[max_i, max_i] = np.cos(fi)
    U[max_j, max_j] = np.cos(fi)
    U[max_j, max_i] = np.sin(fi)
    U[max_i, max_j] = -np.sin(fi)
    return U


def rotate_Jacobi(A):
    U = findU(A).toarray()
    U_T = U.transpose()
    return np.dot(np.dot(U_T, A), U), U


def Jacobi(A, eps):
    old_A = copy.deepcopy(A)
    k = 0
    n, m = A.shape
    U = eye(n, m, format="lil").toarray()
    X = eye(n, m, format="lil").toarray()
    error = eye(n, m)
    while not stop_method(A, eps):
        A, U_new = rotate_Jacobi(A)
        U = np.dot(U, U_new)
        k += 1

    for i in range(n):
        X[:, i] = U[i]
        error = np.dot(old_A, X[i]) - A[i, i] * X[i]

    if k == 0:
        k = 1

    Lambda = np.array([A[i][i] for i in range(n)])

    return Lambda, X, np.linalg.norm(error), k


def print_matrix(A):
    if len(A.shape) == 1:
        n = A.shape[0]
        p = PrettyTable()
        for i in range(n):
            A[i] = np.round(A[i], 5)
        p.add_row(A)
        print(p.get_string(header=False, border=False))
        print()
        return

    n, k = A.shape
    p = PrettyTable()
    for i in range(n):
        for j in range(k):
            A[i][j] = np.round(A[i][j], 5)
        p.add_row(A[i])
    print(p.get_string(header=False, border=False))
    print()


def print_vector(A):
    p = PrettyTable()
    n = len(A)
    for i in range(n):
        A[i] = np.round(A[i], 5)

    p.add_row(A)
    print(p.get_string(header=False, border=False))
    print()


def print_matrix_diagonal(A):
    n, k = A.shape
    p = PrettyTable()
    row = np.zeros(n)
    for i in range(n):
        row[i] = np.round(A[i][i], 5)

    p.add_row(row)
    print(p.get_string(header=False, border=False))
    print()


def print_answer(A, A_new, X, error, conditional_number, iteration, k):
    print("k =", k, "\terror =", round(error, 5), "\tconditional number =", conditional_number, "\titeration =",
          iteration)
    # print("Matrix A:")
    # print_matrix(A)
    # print("Eigenvalues А:")
    # print_vector(A_new)
    # for i in range(k):
    #     print("eigenvector ", i + 1, ":")
    #     print_vector(X[i])


def solve_system(k, generate, eps):
    A = generate(k)
    A_new, X, error, iteration = Jacobi(A, eps)
    conditional_number = get_conditional_number(A)
    return A, A_new, X, error, conditional_number, iteration


def solve_systems(k, generate, eps):
    iterations = []
    conditional_numbers = []
    for i in range(2, k + 1):
        A, A_new, X, error, conditional_number, iteration = solve_system(i, generate, eps)
        iterations.append(iteration)
        conditional_numbers.append(conditional_number)
        print_answer(A, A_new, X, error, conditional_number, iteration, i)

    return iterations, conditional_numbers


# A = np.array([
#     [4, 2, 1],
#     [2, 5, 3],
#     [1, 3, 6]
# ])
#
# A_new, X, error, iterations = Jacobi(A, 0.3)
# print("Matrix A:")
# print_matrix(A)
# print("Eigenvalues А:")
# print_vector(A_new)
# print("eigenvector 1:")
# print_vector(X[0])
# print("eigenvector 2:")
# print_vector(X[1])
# print("eigenvector 3:")
# print_vector(X[2])
# print("error = ", error)
# print("iterations = ", iterations)


# generate_diagonal_matrix
# generate_gilbert_matrix
i, c = solve_systems(40, generate_diagonal_matrix, 0.00001)

b = []
for j in range(len(c)):
    b.append(math.log(i[j], c[j]))

# print(b)
avg = np.average(b[len(c) // 2:])
print(avg)

it = [k for k in range(i[-1])]

arr = []
for k in it:
    arr.append(k ** (0.8 / avg) + 1.1)
plt.plot(it, arr, color="red")

arr = []
for k in it:
    arr.append(k ** (0.99 / avg))
plt.plot(it, arr, color="green")
plt.scatter(i, c)
plt.xlabel("number of iteration")
plt.ylabel("conditional number")
plt.grid(True)
plt.show()
exit()
