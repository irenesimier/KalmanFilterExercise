import numpy as np
from data import RandomTrajectory, TxtTrajectory
from models import ProcessModel, MeasurementModel
from kalman_filter import EKF
from SE2_functions import get_state
from plot import Plot

def main():
    x0 = get_state(theta=0, x=0, y=0)
    R = np.array([[0.01, 0],
                  [0, 0.01]])
    Q = np.array([[0.1, 0, 0],
                  [0, 0.1, 0],
                  [0, 0, 0.1]])
    # Select how trajectory is generated
    data = RandomTrajectory(x0, R, Q, duration=60, dt=0.01)
    #data = TxtTrajectory(R, Q)
    
    P0 = np.eye(3) * 1e-3
    ekf = EKF(x0, P0)

    states = []
    covariances = []
    for i in range(len(data.t)):
        process = ProcessModel(data.noisy_odom[i], data.Q, data.dt)
        measurement = MeasurementModel(data.noisy_gps[i], data.R)
        states.append(ekf.X)
        covariances.append(ekf.P)
        ekf.predict(process)
        ekf.update(measurement)
    states = np.array(states)
    covariances = np.array(covariances)

    Plot(data, states, covariances)
    
if __name__ == "__main__":
    main()
