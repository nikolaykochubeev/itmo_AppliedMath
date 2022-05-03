from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import numpy as np


def jacobi(A, b, tol, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    n = len(A)
    converge = False
    # Create an initial guess if needed
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    while not converge:
        x_new = (b - dot(R, x)) / D
        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= tol
        x = x_new
    return x
