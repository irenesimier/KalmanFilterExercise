import numpy as np
from data import generate_random_trajectory
from models import ProcessModel, MeasurementModel
from kalman_filter import EKF
from SE2_functions import get_state, van_loan_discretization
from plot import Plot

def main():
    x0 = get_state(theta=0, x=0, y=0)
    data = generate_random_trajectory(x0, pos_var=1e-3, v_var=1e-3, w_war=1e-3)

    ekf = EKF(data, process, measurement)

    for t in range(0, int(data.duration / data.dt)):
        process = ProcessModel(data.noisy_odom, data.Q, data.dt)
        measurement = MeasurementModel(data.noisy_gps, data.y, data.R)

        ekf.predict(process)
        ekf.update(measurement)
    
if __name__ == "__main__":
    main()
    Plot.trajectory()
    Plot.params()
