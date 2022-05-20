import numpy as np
from prettytable import PrettyTable


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
