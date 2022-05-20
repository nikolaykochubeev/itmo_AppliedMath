import random
import warnings
from scipy.sparse import eye
import numpy as np
import copy
from prettytable import PrettyTable
from scipy.sparse import csr_matrix as csr


def get_conditional_number(matrixA):
    reverseA = reverse(matrixA).todense()
    reverseA = np.squeeze(np.asarray(reverseA))
    return np.linalg.norm(matrixA) * np.linalg.norm(reverseA)


def LU_decomposition(A):
    warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive. "
                                              "lil_matrix is more efficient.")
    n_rows, n_cols = A.shape
    L = eye(n_rows, n_cols, format="csr")
    U = csr((n_rows, n_cols), dtype=float)
    for i in range(n_rows):
        for j in range(n_cols):
            if i <= j:
                _s = 0
                for k in range(i):
                    _s += L[i, k] * U[k, j]
                U[i, j] = A[i, j] - _s
            else:
                _s = 0
                for k in range(j):
                    _s += L[i, k] * U[k, j]
                L[i, j] = (A[i, j] - _s) / U[j, j]
    return L, U


def getRow(A, i):
    n, m = A.shape
    row = np.zeros(n)
    nzero = A.data[A.indptr[i]:A.indptr[i + 1]]
    inzero = A.indices[A.indptr[i]:A.indptr[i + 1]]
    k = 0
    for i in range(n):
        if k < len(nzero) and i == inzero[k]:
            row[i] = nzero[k]
            k += 1
    return row


def solve(A, b):
    L, U = LU_decomposition(A)
    n, m = A.shape

    y = np.zeros(n)
    for i in range(n):
        row = getRow(L, i)
        y[i] = b[i] - np.dot(y[:i], row[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        row = getRow(U, i)
        x[i] = (y[i] - np.dot(x[i + 1:], row[i + 1:])) / row[i]

    return x


def isNeedToComplete(x_old, x_new):
    eps = 0.0001
    for i in range(len(x_new)):
        if abs(x_old[i] - x_new[i]) > eps:
            return False

    return True


def solution(A, B):
    count = len(B)
    x = np.array([0.0] * count)

    numberOfIter = 0

    while 1:
        x_prev = copy.deepcopy(x)

        for i in range(count):
            S = 0
            for j in range(count):
                if j != i:
                    S = S + A[i, j] * x[j]
            x[i] = B[i] / A[i, i] - S / A[i, i]

        numberOfIter += 1

        if isNeedToComplete(x_prev, x):
            break

    print('Количество итераций на решение: ', numberOfIter * count * (count - 1))

    return x


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


def reverse(A):
    n, m = A.shape
    I = np.eye(n)
    A_reverse = np.zeros(A.shape)
    for i in range(n):
        x = solve(A, I[:, i])
        A_reverse[:, i] = x
    return csr(A_reverse)


def generate_gilbert_matrix(k):
    matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            matrix[i][j] = 1 / (i + j + 1)
    return matrix


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


def solve_system(generator, solver, k):
    A = generator(k)
    x = np.array([i for i in range(1, k + 1)])
    F = np.dot(A, x)
    x_new = solver(A, F)
    F_new = np.dot(A, x_new)
    error = np.linalg.norm(F - F_new)
    conditional_number = get_conditional_number(A)
    return x_new, error, conditional_number


def solve_systems(generator, solver, k):
    error_array = []
    x_new_array = []
    for i in range(2, k + 1):
        x_new, error, conditional_number = solve_system(generator, solver, i)
        x_new_array.append(x_new)
        error_array.append(error)
        print("k =", i, "\terror =", round(error, 5), "\tconditional number =", conditional_number, "\nx' =", *x_new)


# matrixA = np.array([[9.2, 2.5, -3.7],
#               [0.9, 9.0, 0.2],
#               [4.5, -1.6, -10.3],
#               ])
# B = np.array([-17.5, 4.4, -22.1])
# print("Matrix matrixA")
# print_matrix(matrixA)
# print("Matrix B")
# print_matrix(B)
#
# X = solution(matrixA, B)
# print("Matrix X")
# print_matrix(X)
# print("Matrix matrixA * X")
# B1 = np.dot(matrixA, X)
# print_matrix(B1)


A = np.array([
    [6, -3, 5, 0, 2, 0, 0],
    [-4, 0, 7, -3, 0, 2, 0],
    [0, 9, -3, -6, 0, 7, 1],
    [5, -2, 0, 0, 1, 7, -3],
    [-1, 0, 0, 5, 0, 2, 0],
    [9, -8, 7, 0, 2, 3, 0],
    [3, 0, -4, 1, 9, 0, 5]
])
B = np.array([3, 4, 7, 10, 12, 2, 23])
L, U = LU_decomposition(A)
print("Matrix matrixA")
print_matrix(A)
print("Matrix L")
print_matrix(L.toarray())
print("Matrix U")
print_matrix(U.toarray())
A1 = np.dot(L, U)
print("Matrix L * U")
print_matrix(A1.toarray())
A_reverse = reverse(csr(A)).toarray()
print("Reverse Matrix matrixA")
print_matrix(A_reverse)
print("Matrix E")
E = np.dot(A, A_reverse)
print_matrix(E)
X = solve(A, B)
print("Matrix matrixA")
print_matrix(A)
print("Matrix B")
print_matrix(B)
print("Matrix X")
print_matrix(X)
print("Matrix matrixA * X")
B1 = np.dot(A, X)
print_matrix(B1)

# solve_systems(generate_gilbert_matrix, solution, 14)
# solve_systems(generate_diagonal_matrix, solution, 14)
