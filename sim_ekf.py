import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
import math
from system import System
from kalman_filter import KalmanFilter, ExtendedKalmanFilter
from discretization import van_loan_discretization

# --------- SETUP ---------

# Parameters
x0 = np.array([5, 0])  # Initial position (5m), initial velocity (0)
sim_time = 10
dt = 0.01  # Time step

# True system
sys = System(m=1.0, c=0.5, k=5.0, A=5, w=2*math.pi)
solver = RK45(lambda t, x: sys.mass_spring_damper(t, x), 0, x0, sim_time, max_step=dt)

# Discrete-time model for Kalman Filter
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1]])
L = np.array([[0], [1]])
R = 0.05  # Variance of position measurement noise
Q = 0.1  # Variance of accelerometer noise
P0 = np.eye(2) * 0.1
x0_estimate = np.random.multivariate_normal(x0, cov=P0).reshape(2, 1)
Ad, Bd, Qkf = van_loan_discretization(A, B, L, Q, dt)
h = 5
d = 5
kf = ExtendedKalmanFilter(Ad, Bd, Qkf, R, x0_estimate, P0, d, h)

# --------- SIM ---------

times = []
true_position = []
true_velocity = []
measured_position = []
filtered_position = []
filtered_velocity = []
position_error = []
velocity_error = []
position_bounds = []
velocity_bounds = []

counter = 0
# Simulate the system and generate noisy measurements
while solver.t < sim_time:
    solver.step()
    times.append(solver.t)
    
    # True position and velocity
    state = solver.y.reshape(2, 1)
    true_position.append(state[0])
    true_velocity.append(state[1])
    
    # Get sensor data
    measured_dist = sys.distance_from_wall(state, d, h, R)
    measured_pos = np.sqrt(np.abs(measured_dist**2 - h**2)) - d
    measured_position.append(measured_pos)
    measured_accel = sys.noisy_acceleration(solver.t, state, Q)
    
    # Kalman Filter
    kf.predict(measured_accel)
    kf.update(measured_dist)
    filtered_state = kf.get_state()
    filtered_position.append(filtered_state[0])
    filtered_velocity.append(filtered_state[1])
    
    # Error
    position_error.append(filtered_state[0] - state[0])
    velocity_error.append(filtered_state[1] - state[1])
    std_bounds = kf.get_covariance()
    position_bounds.append(math.sqrt(std_bounds[0, 0]) * 3)
    velocity_bounds.append(math.sqrt(std_bounds[1, 1]) * 3)

times = np.array(times)
true_position = np.array(true_position)
true_velocity = np.array(true_velocity)
measured_position = np.array(measured_position)
filtered_position = np.array(filtered_position)
filtered_velocity = np.array(filtered_velocity)
position_error = np.array(position_error)
velocity_error = np.array(velocity_error)
position_bounds = np.array(position_bounds)
velocity_bounds = np.array(velocity_bounds)

# --------- PLOTS ---------

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Position plot
axs[0, 0].plot(times, true_position, label="True Position", color='blue')
axs[0, 0].plot(times, measured_position, label="Measured Position", color='orange', linestyle='--')
axs[0, 0].plot(times, filtered_position, label="Filtered Position", color='green', linestyle='-.')
axs[0, 0].set_title("Position of the Mass-Spring-Damper System")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Position [m]")
axs[0, 0].grid(True)
axs[0, 0].legend()

# Velocity plot
axs[0, 1].plot(times, true_velocity, label="True Velocity", color='blue')
axs[0, 1].plot(times, filtered_velocity, label="Filtered Velocity", color='green', linestyle='-.')
axs[0, 1].set_title("Velocity of the Mass-Spring-Damper System")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Velocity [m/s]")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Position error plot
axs[1, 0].plot(times, position_error, label="Position Error", color='red')
axs[1, 0].plot(times, +position_bounds, label="+3σ Bound", color='gray', linestyle='--')
axs[1, 0].plot(times, -position_bounds, label="-3σ Bound", color='gray', linestyle='--')
axs[1, 0].set_title("Position Estimation Error with ±3σ Bounds")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Error [m]")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Velocity error plot
axs[1, 1].plot(times, velocity_error, label="Velocity Error", color='red')
axs[1, 1].plot(times, +velocity_bounds, label="+3σ Bound", color='gray', linestyle='--')
axs[1, 1].plot(times, -velocity_bounds, label="-3σ Bound", color='gray', linestyle='--')
axs[1, 1].set_title("Velocity Estimation Error with ±3σ Bounds")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Error [m/s]")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()