from matrix import *


def linear_solve_without(A: Matrix, b):
    """x = linear_solve_without_pivoting(A, b) is the solution to A x = b (computed without pivoting)
       A is any matrix
       b is a vector of the same leading dimension as A
       x will be a vector of the same leading dimension as A
    """
    (L, U) = A.lu_decomposition()
    x, iters = lu_solve(L, U, b)
    return x, iters


def lu_solve(L: LMatrix, U: UMatrix, b):
    """x = lu_solve(L, U, b) is the solution to L U x = b
       L must be a lower-triangular matrix
       U must be an upper-triangular matrix of the same size as L
       b must be a vector of the same leading dimension as L

       x = linear_solve_without_pivoting(A, b) is the solution to A x = b (computed without pivoting)
       A is any matrix
       b is a vector of the same leading dimension as A
       x will be a vector of the same leading dimension as A
    """
    y, iters1 = forward_sub(L, b)
    x, iters2 = back_sub(U, y)
    return x, iters1 + iters2


def forward_sub(L: LMatrix, b):
    """x = forward_sub(L, b) is the solution to L x = b
       L must be a lower-triangular matrix
       b must be a vector of the same leading dimension as L
    """
    n = len(L[0])
    x = np.zeros(n)
    iters = 0
    for i in range(n):
        tmp = b[i]
        for j in range(i - 1):
            iters += 1
            tmp -= L[i][j] * x[j]
        x[i] = tmp / L[i][i]
    return x, iters


def back_sub(U: UMatrix, b):
    """x = back_sub(U, b) is the solution to U x = b
       U must be an upper-triangular matrix
       b must be a vector of the same leading dimension as U
    """
    n = len(U[0])
    x = np.zeros(n)
    iters = 0
    for i in range(n - 1, -1, -1):
        tmp = b[i]
        for j in range(i + 1, n):
            iters += 1
            tmp -= U[i][j] * x[j]
        x[i] = tmp / U[i][i]
    return x, iters


if __name__ == '__main__':
    m = Matrix.from_data([[4, -1, -1],
                          [-2, 6, 1],
                          [-1, 1, 7]])
    b = [3, 9, -6]
    print("numpy solve: ", np.linalg.solve(m, b))
    print("our solve: ", lu_solve(m, b))

    A2 = Matrix.from_data([[10., -1., 2., 0.],
                           [-1., 11., -1., 3.],
                           [2., -1., 10., -1.],
                           [0., 3., -1., 8.]])
    b2 = np.array([6.0, 25.0, -11.0, 15.0])

    print("numpy solve: ", np.linalg.solve(A2, b2))
    print("our solve: ", lu_solve(A2, b2))