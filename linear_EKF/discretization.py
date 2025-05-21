import numpy as np
from scipy.linalg import expm

def van_loan_discretization(A, B, L, Qc, dt):
    n = A.shape[0]
    m = B.shape[1] # nb of inputs

    # Step 1: Compute Ad and Bd using Zero-Order Hold (ZOH)
    M_zoh = np.block([
        [A, B],
        [np.zeros((m, n + m))]
    ]) * dt

    expM = expm(M_zoh)
    Ad = expM[:n, :n]
    Bd = expM[:n, n:]

    # Step 2: Compute Qk via Van Loan
    M_vl = np.block([
        [-A, (L * Qc) @ L.T],
        [np.zeros((n, n)), A.T]
    ]) * dt

    Phi = expm(M_vl)
    Phi12 = Phi[:n, n:]
    Phi22 = Phi[n:, n:]
    Qk = Ad @ Phi12

    return Ad, Bd, Qk
