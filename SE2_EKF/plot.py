import matplotlib.pyplot as plt
import numpy as np
from pymlg import SE2
from SE2_functions import get_angle

class Plot:
    def __init__(self, data, states, covariances):
        self.data = data
        self.states = states
        self.covariances = covariances

        self.trajectory()
        self.params()
        plt.show()

    def trajectory(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.data.gt_states[:, 0, 2], self.data.gt_states[:, 1, 2], label="True Trajectory", linewidth=2)
        plt.plot(self.states[:, 0, 2], self.states[:, 1, 2], label="Estimated Trajectory", linestyle='--')
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("Trajectory Comparison")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

    def params(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))

        # Theta
        gt_theta = [get_angle(mat) for mat in self.data.gt_states]
        est_theta = [get_angle(mat) for mat in self.states]
        axs[0, 0].plot(self.data.t, gt_theta, label="True θ", linewidth=2)
        axs[0, 0].plot(self.data.t, est_theta, label="Estimated θ", linestyle='--')
        axs[0, 0].set_title("Theta")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # X
        axs[1, 0].plot(self.data.t, self.data.gt_states[:, 0, 2], label="True x", linewidth=2)
        axs[1, 0].plot(self.data.t, self.states[:, 0, 2], label="Estimated x", linestyle='--')
        axs[1, 0].set_title("X Position")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Y
        axs[2, 0].plot(self.data.t, self.data.gt_states[:, 1, 2], label="True y", linewidth=2)
        axs[2, 0].plot(self.data.t, self.states[:, 1, 2], label="Estimated y", linestyle='--')
        axs[2, 0].set_title("Y Position")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Error Plots
        d_xi = []
        sigma3 = []

        for i in range(len(self.data.gt_states)):
            error = SE2.inverse(self.data.gt_states[i]) @ self.states[i]
            #error = SE2.inverse(self.states[i]) @ self.data.gt_states[i]
            d_xi.append(SE2.vee(SE2.log(error)))

            # Get diagonal entries (variances) and compute ±3σ
            diag_vars = np.diag(self.covariances[i])  # shape (3,)
            sigma3.append(3 * np.sqrt(diag_vars))     # shape (3,)

        d_xi = np.array(d_xi)        # shape (N, 3, 1)
        sigma3 = np.array(sigma3)    # shape (N, 3)

        # Plot error and ±3σ bounds
        axs[0, 1].plot(self.data.t, d_xi[:, 0, 0], color='red', label='Error')
        axs[0, 1].plot(self.data.t, +sigma3[:, 0], 'k--', label='+3σ')
        axs[0, 1].plot(self.data.t, -sigma3[:, 0], 'k--', label='-3σ')
        axs[0, 1].set_title("Error in Theta")
        axs[0, 1].grid(True)

        axs[1, 1].plot(self.data.t, d_xi[:, 1, 0], color='red', label='Error')
        axs[1, 1].plot(self.data.t, +sigma3[:, 1], 'k--', label='+3σ')
        axs[1, 1].plot(self.data.t, -sigma3[:, 1], 'k--', label='-3σ')
        axs[1, 1].set_title("Error in X")
        axs[1, 1].grid(True)

        axs[2, 1].plot(self.data.t, d_xi[:, 2, 0], color='red', label='Error')
        axs[2, 1].plot(self.data.t, +sigma3[:, 2], 'k--', label='+3σ')
        axs[2, 1].plot(self.data.t, -sigma3[:, 2], 'k--', label='-3σ')
        axs[2, 1].set_title("Error in Y")
        axs[2, 1].grid(True)

        for ax in axs.flat:
            ax.set_xlabel("t")
            ax.legend()

        plt.tight_layout()