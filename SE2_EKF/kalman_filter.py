class EKF:
    def predict(self, process):
        self.X = 
        self.P = 


    def update(self, measurement):
        C = measurement.jacobian(self.X)
        S = C @ self.P @ C.T + measurement.R
        K = self.P @ C.T @ np.linalg.inv(S)
        d_eps = K @ (measurement.y - measurement.evaluate(self.X))
        self.X = self.X @ SE2.Exp(-d_eps)
        self.P = (np.eye(self.P.shape[0]) - K @ C) @ self.P