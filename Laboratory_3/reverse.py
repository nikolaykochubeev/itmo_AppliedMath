import numpy as np
from scipy.sparse import csr_matrix as csr
from LU_Solution import solve

def reverse(A):
    n, m = A.shape
    I = np.eye(n)
    A_reverse = np.zeros(A.shape)
    for i in range(n):
        x = solve(A, I[:, i])
        A_reverse[:, i] = x
    return csr(A_reverse)