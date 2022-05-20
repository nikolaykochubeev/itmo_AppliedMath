import numpy as np
from scipy.sparse import csr_matrix as csr
from scipy.sparse import eye
import random
from prettytable import PrettyTable
from print_Matrix import print_matrix
from LU_Decomposition import LU_decomposition
from reverse import reverse
from LU_Solution import solve

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
# L, U = LU_decomposition(A)
# print("Matrix A")
# print_matrix(A)
# print("Matrix L")
# print_matrix(L.toarray())
# print("Matrix U")
# print_matrix(U.toarray())
# B = np.dot(L, U)
# print("Matrix L * U")
# print_matrix(B.toarray())
#
# A_reverse = reverse(csr(A)).toarray()
# print("Reverse Matrix A")
# print_matrix(A_reverse)
# print("Matrix E")
# E = np.dot(A, A_reverse)
# print_matrix(E)

X = solve(A, B)
print("Matrix A")
print_matrix(A)
print("Matrix B")
print_matrix(B)
print("Matrix X")
print_matrix(X)
print("Matrix A * X")
B1 = np.dot(A, X)
print_matrix(B1)
exit()