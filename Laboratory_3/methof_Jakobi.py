import numpy as np
from scipy.sparse import csr_matrix as csr
import random
import math
import copy
from scipy.sparse import eye
from print_Matrix import print_matrix
from solve_gilbert import solve_systems, solve_system, generate_gilbert_matrix


def stop_method(A):
    n, m = A.shape
    eps = 0.0001
    s = 0
    for j in range(n):
        for i in range(j):
            s += A[i, j] * A[i, j]

    return s < eps


def findMax(A):
    n, m = A.shape
    max = -1
    max_i = -1
    max_j = -1
    for j in range(n):
        for i in range(j):
            if max < abs(A[i, j]):
                max = A[i, j]
                max_i = i
                max_j = j

    return max, max_i, max_j


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


A = np.array([
    [4, 2, 1],
    [2, 5, 3],
    [1, 3, 6]
])


def rotate_Jacobi(A):
    U = findU(A).toarray()
    U_T = U.transpose()
    return np.dot(np.dot(U_T, A), U), U


def Jacobi(A):
    old_A = copy.deepcopy(A)
    k = 0
    n, m = A.shape
    U = eye(n, m, format="lil").toarray()
    X = eye(n, m, format="lil").toarray()
    while not stop_method(A):
        A, U_new = rotate_Jacobi(A)
        U = np.dot(U, U_new)
        k += 1

    for i in range(n):
        X[:, i] = np.dot(old_A, U[:, i]) - A[i, i] * U[:, i]

    print_matrix(X)


Jacobi(A)
exit()
