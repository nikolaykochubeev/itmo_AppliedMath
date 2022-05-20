import numpy as np
import copy
from print_Matrix import print_matrix


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

A = np.array([[9.2, 2.5, -3.7],
              [0.9, 9.0, 0.2],
              [4.5, -1.6, -10.3],
              ])
B = np.array([-17.5, 4.4, -22.1])

# print("Matrix A")
# print_matrix(A)
# print("Matrix B")
# print_matrix(B)
#
# X = solution(A, B)
# print("Matrix X")
# print_matrix(X)
# print("Matrix A * X")
# B1 = np.dot(A, X)
# print_matrix(B1)
