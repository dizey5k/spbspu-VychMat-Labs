import numpy as np
from scipy.linalg import lu_solve

def compute_inverse(lu, piv):
    n = lu.shape[0]
    invA = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        invA[:, i] = lu_solve((lu, piv), e)
    return invA