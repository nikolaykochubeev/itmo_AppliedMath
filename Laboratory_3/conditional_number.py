import numpy as np
from reverse import reverse


def get_conditional_number(A):
    A_reverse = reverse(A).todense()
    A_reverse = np.squeeze(np.asarray(A_reverse))
    return np.linalg.norm(A) * np.linalg.norm(A_reverse)