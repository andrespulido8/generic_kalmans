import numpy as np

class KalmanFilter:
    """Multivariante Kalman Filter implementation."""

    def __init__(self, P, Q, R, X=np.array([0, 0, -0.02, -0.01])):
        """Initialize the Kalman Filter.
        Parameters
        :param H: Measurement matrix
        :param Q: Process noise covariance matrix
        :param R: Measurement noise covariance matrix
        :param K: Kalman gain
        :param P: State Covariance matrix
        :param X: State vector
        """
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.H = H
        self.Q = Q
        self.R = R
        self.K = np.zeros((np.shape(H)[1], np.shape(H)[0]))
        self.P = P 
        self.X = X
        self.last_update_time = None

    def predict(self, dt):
        """Function that predicts the state and covariance of the system.
        Parameters
        :param X: State vector
        :param P: Covariance matrix
        :param Q: Process noise covariance matrix
        :param dt: Time step
        :param F: State Transition matrix
        Returns
        :return: X_pred, P_pred"""

        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )  # State Transition matrix
        self.X = F @ self.X  # Predicting the State
        self.P = F @ self.P @ F.T + self.Q  # Predicting the Covariance

    def update(self, z, R=None):
        """Function that updates the state and covariance of the system.
        Parameters
        :param z: Measurement vector
        :param X_pred: Predicted state vector
        :param P_pred: Predicted covariance matrix
        :param H: Measurement matrix
        :param R: Measurement noise covariance matrix
        Returns
        :return: X, P"""
        if R is not None:
            R = self.R
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.X = self.X + K @ y
        self.P = (np.eye(np.shape(K)[0]) - K @ self.H) @ self.P
