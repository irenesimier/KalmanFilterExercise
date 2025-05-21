import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
import math
from system import System

# Parameters for the system
m = 1.0     # Mass (kg)
c = 0.5     # Damping coefficient (kg/s)
k = 5.0     # Spring constant (N/m)
A = 5       # rad/s
w = math.pi # m/s

# Set up solver
x0 = [5.0, 0.0] # Initial position (5m), initial velocity (0)cdh    
sim_time = 10
sys = System(m=1.0, c=0.5, k=5.0, A=5, w=2*math.pi)
solver = RK45(lambda t, y: sys.mass_spring_damper(t, y), 0, x0, sim_time, max_step=0.01)

# Solve
times = []
forces = []
positions = []
velocities = []
while solver.t < sim_time:
    solver.step()
    times.append(solver.t)
    forces.append(sys.ext_force(solver.t))
    positions.append(solver.y[0])
    velocities.append(solver.y[1])
times = np.array(times)
positions = np.array(positions)
velocities = np.array(velocities)
forces = np.array(forces)

# Plot the results
plt.figure(figsize=(10, 6))

# Position vs Time
plt.subplot(3, 1, 1)
plt.plot(times, positions, label="Position")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.grid(True)
plt.legend()

# Velocity vs Time
plt.subplot(3, 1, 2)
plt.plot(times, velocities, label="Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.grid(True)
plt.legend()

# External Force vs Time
plt.subplot(3, 1, 3)
plt.plot(times, forces, label="External Force")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

