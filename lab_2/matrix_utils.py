import numpy as np

# infinity matrix norm
def norm_inf_matrix(A):
    n = A.shape[0]
    max_sum = 0.0
    for i in range(n):
        row_sum = np.sum(np.abs(A[i]))
        if row_sum > max_sum:
            max_sum = row_sum
    return max_sum

# max abs elem
def norm_inf_vector(x):
    return np.max(np.abs(x))

def cond_matrix(A, invA):
    return norm_inf_matrix(A) * norm_inf_matrix(invA)