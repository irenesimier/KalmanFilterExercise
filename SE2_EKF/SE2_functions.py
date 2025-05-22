import numpy as np
from scipy.linalg import expm
from pymlg.numpy import SO2, SE2

def get_state(theta, x, y):
#     c = np.cos(theta)
#     s = np.sin(theta)
#     return np.array([
#         [c, -s, x],
#         [s,  c, y],
#         [0,  0, 1]
#     ])
    return SE2.from_components(SO2.Exp(theta), np.array([x, y]))

def get_angle(state):
        theta = np.arctan2(state[1, 0], state[0, 0])
        return theta

def van_loan_discretization(A, B, L, Qc, dt):
    n = A.shape[0]
    m = B.shape[1]

    # Compute Ad and Bd using ZOH
    M_zoh = np.block([
        [A, B],
        [np.zeros((m, n + m))]
    ]) * dt

    expM = expm(M_zoh)
    Ad = expM[:n, :n]
    Bd = expM[:n, n:]

    # Compute Qk with Van Loan
    M_vl = np.block([
        [-A, (L * Qc) @ L.T],
        [np.zeros((n, n)), A.T]
    ]) * dt

    Phi = expm(M_vl)
    Phi12 = Phi[:n, n:]
    Phi22 = Phi[n:, n:]
    Qk = Ad @ Phi12

    return Ad, Bd, Qk