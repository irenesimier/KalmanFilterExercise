import numpy as np
import math
from scipy.signal import cont2discrete

class System:
    def __init__(self, m, c, k, A, w):
        self.m = m
        self.c = c
        self.k = k
        self.A = A
        self.w = w

    def ext_force(self, t):
        return self.A * math.sin(self.w * t)

    def acceleration(self, t, r, v):
        return (1 / self.m) * (self.ext_force(t) - self.k * r - self.c * v)

    def mass_spring_damper(self, t, x):
        """
        Input: x = [position, velocity]
        Output: dx/dt = [velocity, acceleration]
        """
        dxdt = np.zeros_like(x)
        dxdt[0] = x[1]  # velocity
        dxdt[1] = self.acceleration(t, x[0], x[1])  # acceleration
        return dxdt
    
    def noisy_acceleration(self, t, x, Q):
        true_acc = self.acceleration(t, x[0], x[1])
        noise = np.random.normal(0, np.sqrt(Q), 1)
        uacc = true_acc + noise
        return uacc

    def noisy_position(self, x, R):
        noise = np.random.normal(0, np.sqrt(R), 1)
        y = x[0] + noise
        return y
    
    def distance_from_wall(self, x, d, h, R):
        noise = np.random.normal(0, np.sqrt(R), 1)
        y = math.sqrt((d+x[0])**2 + h**2) + noise
        return y
