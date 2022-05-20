import numpy as np
from reverse import reverse

def norm(A):
    result = 0
    k, n = A.shape
    for i in range(k):
        for j in range(n):
            result += A[i][j] ** 2
    return np.sqrt(result)

def get_conditional_number(A):
    A_reverse = reverse(A).todense()
    A_reverse = np.squeeze(np.asarray(A_reverse))
    return norm(A) * norm(A_reverse)