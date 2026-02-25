import numpy as np
from scipy.linalg import lu_factor, lu_solve
from data import get_matrix_and_rhs
from matrix_utils import norm_inf_vector, cond_matrix
from solver import compute_inverse

def main():
    p_values = [1.0, 0.1, 0.01, 0.0001, 0.000001]

    print("===========================================================================")
    print("compare solutions: x1 = A^(-1)*b И x2 fro diff p")
    print("===========================================================================")
    print("p          cond(A)    sigma = ||x1-x2||/||x1||")
    print("---        ------     -------------------------")

    for p in p_values:
        try:
            A, b = get_matrix_and_rhs(p)

            lu, piv = lu_factor(A)

            invA = compute_inverse(lu, piv)

            cond = cond_matrix(A, invA)

            # x1 = A^(-1) * b
            x1 = invA @ b

            x2 = lu_solve((lu, piv), b)

            diff_norm = norm_inf_vector(x1 - x2)
            x1_norm = norm_inf_vector(x1)
            sigma = diff_norm / x1_norm if x1_norm != 0 else float('inf')

            print(f"{p:8.6f}   {cond:8.2e}   {sigma:8.2e}")

        except Exception as e:
            print(f"err for p={p}: {e}")

    print("===========================================================================")

if __name__ == "__main__":
    main()