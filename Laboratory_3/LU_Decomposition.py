from scipy.sparse import csr_matrix as csr
from scipy.sparse import eye

def LU_decomposition(A):
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

