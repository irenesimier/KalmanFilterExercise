import numpy as np
from scipy.linalg import expm

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A = A  
        self.B = B    
        self.C = C
        self.Q = Q 
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, measured_accel):
        self.x = self.A @ self.x + self.B * measured_accel
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measured_pos):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + K @ (measured_pos - self.C @ self.x)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.C) @ self.P @ (I - K @ self.C).T + K * self.R @ K.T

    def get_state(self):
        return self.x
    
    def get_covariance(self):
        return self.P

class ExtendedKalmanFilter:
    def __init__(self, A, B, Q, R, x0, P0, d, h):
        self.A = A  
        self.B = B    
        self.Q = Q 
        self.R = R
        self.x = x0
        self.P = P0
        self.d = d
        self.h = h

    def predict(self, measured_accel):
        self.x = self.A @ self.x + self.B * measured_accel
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measured_dist):
        dgdr = float((self.d + self.x[0]) / np.sqrt((self.d + self.x[0])**2 + self.h**2))
        C = np.array([[dgdr, 0]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        self.x = self.x + K * (measured_dist - np.sqrt((self.d + self.x[0])**2 + self.h**2))
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ C) @ self.P @ (I - K @ C).T + K * self.R @ K.T

    def get_state(self):
        return self.x
    
    def get_covariance(self):
        return self.P
