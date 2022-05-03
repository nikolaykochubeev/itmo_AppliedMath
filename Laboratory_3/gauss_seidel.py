import numpy as np


def gauss_seidel_solve(A, b: list, iters_limit: int = 10000000000, tol: float = 1e-3):
    n = len(A)
    x = np.zeros(n)  # zero vector
    converge = False
    iter = 1

    print("System of equations:")
    for i in range(n):
        row = ["{0:3g}*x{1}".format(A[i][j], j + 1) for j in range(len(A[1]))]
        print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

    while not converge:
        x_new = np.copy(x)
        if iter > iters_limit:
            raise IndexError
        print("Iteration {0}: {1}".format(iter, x_new))

        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        iter += 1
        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= tol
        x = x_new

    print("final solve: {0}".format(x))
    return x.tolist(), iter