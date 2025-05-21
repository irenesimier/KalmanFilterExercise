import numpy as np
from pymlg import SE2
from SE2_functions import van_loan_discretization

class ProcessModel:
    def __init__(self, u, Q: np.ndarray, dt):
        self.dt = dt
        self.u = u * self.dt
        self.Q = Q # continuous

    def evaluate(self, state):
        return state @ SE2.Exp(self.u)

    def covariance(self, P):
        A = SE2.adjoint(SE2.Exp(self.u))
        L = - SE2.left_jacobian(- self.u)
        _, _, Qd = van_loan_discretization(A, L, L, self.Q, self.dt)
        P = A @ P @ A.T + L @ Qd @ L.T
        return P


class MeasurementModel:
    def __init__(self, y, R: np.ndarray):
        self.y = y
        self.R = R

    def evaluate(self, state):
        return np.array([state[0, 2], state[1, 2]]).reshape(2,1)
    
    def jacobian(self, state):
        return - np.array([[0, state[0, 0], state[0, 1]],
                           [0, state[1, 0], state[1, 1]]])
