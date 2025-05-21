import numpy as np
import random
import matplotlib.pyplot as plt
import time
from pymlg import SE2
from SE2_functions import get_state

class RandomTrajectory:
    def __init__(self, x0=get_state(0,0,0), pos_var=1e-3, v_var=1e-3, w_var=1e-6, duration=60, dt=0.01):
        # Sim params
        self.dt = dt
        self.t = np.arange(0, duration, dt)

        # State Params
        self.gt_states = [x0]

        # Measurements
        self.gps = None
        self.odom = None
        self.noisy_gps = None
        self.noisy_odom = None

        # Covariance
        self.R = np.array([[pos_var, 0.0],[0.0, pos_var]])
        self.Q = np.sqrt(np.diag([self.dt*w_var, self.dt*v_var, self.dt*v_var])) # continuous-time variances

        # Generate data
        self.generate_data()

    def generate_data(self):
            
        N = len(self.t)

        # Create smooth velocity and yaw rate profiles
        v = random.randint(1, 5) + self.smooth_random_signal(scale=5, cutoff_freq=0.1)  # m/s
        v = np.clip(v, 0.1, 10.0)  # avoid going backwards or too fast
        w = random.randint(0, 10) * 0.01 + self.smooth_random_signal(scale=0.2, cutoff_freq=0.05)  # rad/s

        # odom
        self.odom = np.stack([
            np.array([w[i], v[i], 0]).reshape(3, 1)
            for i in range(N)
        ])
        self.noisy_odom = np.stack([
            np.random.multivariate_normal(mean=self.odom[i].flatten(), cov=self.Q).reshape((3, 1))
            for i in range(N)
        ])

        # Ground truth states
        for i in range(1, N):
            self.gt_states.append(self.gt_states[-1] @ SE2.Exp(self.odom[i-1] * self.dt))
        shifted_gt_states = self.gt_states[1:].copy()
        shifted_gt_states.append(shifted_gt_states[-1] @ SE2.Exp(self.odom[i-1] * self.dt))
        self.gt_states = np.stack(self.gt_states)  

        # GPS
        self.gps = np.stack([
            np.array([[T[0, 2]], [T[1, 2]]])
            for T in shifted_gt_states
        ])
        self.noisy_gps = np.stack([
            np.random.multivariate_normal(mean=self.gps[i].flatten(), cov=self.R).reshape(2, 1)
            for i in range(N)
        ])

    def smooth_random_signal(self, scale=1.0, cutoff_freq=0.1):
        """
        Generates a smooth random signal using filtered noise.
        """

        N = len(self.t)
        white_noise = np.random.randn(N) * scale
        # Simple low-pass filter using convolution
        window_size = int(1 / (cutoff_freq * self.dt))
        window = np.exp(-np.linspace(-2, 2, window_size)**2)
        window /= np.sum(window)
        smooth_signal = np.convolve(white_noise, window, mode='same')
        return smooth_signal

    def plot_trajectory(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.gps[:, 0], self.gps[:, 1], label='Ground Truth Trajectory', linewidth=2)
        plt.plot(self.noisy_gps[:, 0], self.noisy_gps[:, 1], label='Noisy Position', linestyle='--', alpha=0.7)
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Trajectory')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_velocities(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Linear velocity
        axs[0].plot(self.t, self.odom[:, 1], label='True Linear Velocity', color='blue')
        axs[0].plot(self.t, self.noisy_odom[:, 1], label='Noisy Linear Velocity', linestyle='--', color='skyblue')
        axs[0].set_ylabel("Linear Velocity (m/s)")
        axs[0].grid(True)
        axs[0].legend()

        # Angular velocity
        axs[1].plot(self.t, self.odom[:, 0], label='True Angular Velocity', color='orange')
        axs[1].plot(self.t, self.noisy_odom[:, 0], label='Noisy Angular Velocity', linestyle='--', color='gold')
        axs[1].set_ylabel("Angular Velocity (rad/s)")
        axs[1].set_xlabel("Time (s)")
        axs[1].grid(True)
        axs[1].legend()

        plt.suptitle("Velocities (True vs. Noisy)")
        plt.show()

if __name__ == "__main__":
    traj = RandomTrajectory()
    traj.plot_trajectory()
    traj.plot_velocities()