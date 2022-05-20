import numpy as np
from LU_Decomposition import LU_decomposition

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