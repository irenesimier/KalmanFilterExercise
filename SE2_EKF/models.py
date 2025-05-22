import numpy as np
from pymlg import SE2
from SE2_functions import van_loan_discretization

class ProcessModel:
    def __init__(self, u, Q: np.ndarray, dt):
        self.dt = dt
        self.u = u
        self.Q = Q # continuous

    def evaluate(self, state):
        return state @ SE2.Exp(self.u * self.dt)

    def covariance(self, P):
        A = SE2.adjoint(SE2.Exp(- self.u))
        L = - SE2.left_jacobian(- self.u)
        Ad, Ld, Qd = van_loan_discretization(A, L, L, self.Q, self.dt)
        P = Ad @ P @ Ad.T + Ld @ Qd @ Ld.T
        return P


class MeasurementModel:
    def __init__(self, y, R: np.ndarray):
        self.y = y
        self.R = R

    def evaluate(self, state):
        return np.array([state[0, 2], state[1, 2]]).reshape(2,1)
    
    def jacobian(self, state):
        jac = - np.array([[0, state[0, 0], state[0, 1]],
                                [0, state[1, 0], state[1, 1]]])
        jac_fd = - self.jacobian_fd(state)    
        assert np.allclose(jac_fd, jac, atol=1e-6)
        
        return jac
    

    def jacobian_fd(self, state, step_size=1e-6):
        """
        Used to check jacobian above
        """
        N = state.shape[0]
        y = self.evaluate(state)
        m = y.size
        jac_fd = np.zeros((m, N))
        for i in range(N):
            dx = np.zeros((N,))
            dx[i] = step_size
            x_temp = state @ SE2.Exp(dx)
            jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / step_size

        return jac_fd