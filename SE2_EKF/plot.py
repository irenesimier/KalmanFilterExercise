import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self, data, est_x, est_y, est_theta):
        self.data = data
        self.est_x = est_x
        self.est_y = est_y
        self.est_theta = est_theta

    def trajectory(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.true_x, self.data.y, label="True Trajectory", linewidth=2)
        plt.plot(self.est_x, self.est_y, label="Estimated Trajectory", linestyle='--')
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("Trajectory Comparison")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def params(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))

        # Theta
        axs[0, 0].plot(self.data.time, self.data.theta, label="True θ", linewidth=2)
        axs[0, 0].plot(self.data.time, self.est_theta, label="Estimated θ", linestyle='--')
        axs[0, 0].set_title("Theta")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # X
        axs[1, 0].plot(self.data.time, self.true_x, label="True x", linewidth=2)
        axs[1, 0].plot(self.data.time, self.est_x, label="Estimated x", linestyle='--')
        axs[1, 0].set_title("X Position")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Y
        axs[2, 0].plot(self.data.time, self.data.y, label="True y", linewidth=2)
        axs[2, 0].plot(self.data.time, self.est_y, label="Estimated y", linestyle='--')
        axs[2, 0].set_title("Y Position")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Error Plots
        axs[0, 1].plot(self.data.time, np.array(self.data.theta) - np.array(self.est_theta), color='red')
        axs[0, 1].set_title("Error in Theta")
        axs[0, 1].grid(True)

        axs[1, 1].plot(self.data.time, np.array(self.true_x) - np.array(self.est_x), color='red')
        axs[1, 1].set_title("Error in X")
        axs[1, 1].grid(True)

        axs[2, 1].plot(self.data.time, np.array(self.data.y) - np.array(self.est_y), color='red')
        axs[2, 1].set_title("Error in Y")
        axs[2, 1].grid(True)

        for ax in axs.flat:
            ax.set_xlabel("Time")
        plt.tight_layout()
        plt.show()
