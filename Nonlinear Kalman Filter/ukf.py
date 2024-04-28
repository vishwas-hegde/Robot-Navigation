import numpy as np
from math import cos, sin
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation
from typing import Tuple
import matplotlib.pyplot as plt
import task3
from scipy.linalg import block_diag

class UKF:
    def __init__(self, P, Q, R, dt):
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
        self.n = 15
        self.m = 15
        self.kappa = 1
        self.alpha = 0.01
        self.beta = 2

    def compute_sigma(self, x, P):
        """
        Generate sigma points.
        Input:
            x : mean.
            P : covariance matrix.
        Return:
            Sigma Points.
        """
        size = x.shape[0]
        n = self.n
        sigma_points = np.zeros((2 * n + 1, size))
        sigma_points[0] = x

        l = self.alpha ** 2 * (size + self.kappa) - size # scaling factor
        try:
            cov = sqrtm((n + l) * P)
        except:
            print("error")
            print(P)

        for i in range(1, n + 1):
            sigma_points[i] = x + cov[i - 1]
            sigma_points[n + i] = x - cov[i - 1]

        return sigma_points.T

    def compute_weights(self, x):
        """
        Compute weights.
        Input:
            x : mean.
            P : covariance matrix.
        Return:
            Weights.
        """
        size = x.size
        l = self.alpha ** 2 * (size + self.kappa) - size  # scaling factor
        W_mean = np.zeros(2 * size + 1) + 1 / (2 * (l + size))
        W_cov = np.zeros(2 * size + 1) + 1 / (2 * (l + size))

        W_mean[0] = l / (size + l)
        W_cov[0] = l / (size + l) + (1 - self.alpha ** 2 + self.beta)

        return W_mean, W_cov


    def compute_state(self,x,u):
        """
        Compute the state.
        Input:
            x : state.
            u : imu data.
        Return:
            xdot : state derivative.
        """
        phi, theta, psi = x[3:6]

        R = np.zeros((3, 3))
        R[0, 0] = cos(psi) * cos(theta) - sin(psi) * sin(theta) * sin(phi)
        R[0, 1] = -cos(phi) * sin(psi)
        R[0, 2] = cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi)
        R[1, 0] = cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)
        R[1, 1] = cos(psi) * cos(phi)
        R[1, 2] = sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi)
        R[2, 0] = -cos(phi) * sin(theta)
        R[2, 1] = sin(phi)
        R[2, 2] = cos(phi) * cos(theta)

        G = np.zeros((3, 3))
        G[0, 0] = cos(theta)
        G[0, 2] = -cos(phi) * sin(theta)
        G[1, 1] = 1
        G[1, 2] = sin(phi)
        G[2, 0] = sin(theta)
        G[2, 2] = cos(phi) * cos(theta)

        xdot = np.zeros(x.shape)
        xdot[0:3] = x[6:9]
        xdot[3:6] = np.linalg.inv(G) @ (u[0:3] - x[9:12])
        xdot[6:9] = np.array([0, 0, -9.81]) + R @ (u[3:6] - x[12:15])

        return xdot


    def calculate_observation(self,x: np.ndarray) -> np.ndarray:
        """
        Calculate the state observation.

        Args:
            x (np.ndarray): The current state.

        Returns:
            np.ndarray: The observation.
        """
        # observation matrix
        C = np.zeros((6, 15))
        C[0:6, 0:6] = np.eye(6, 6)

        return C @ x


    def predict(self,x: np.ndarray,
                u: np.ndarray,
                P: np.ndarray,
                Q: np.ndarray,
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the next state.
        Input:
            x : initial state.
            u : imu, gyroscope input.
            P : initial state covariance matrix.
            Q : process noise.
            dt : time step.

        Return:
            predicted mean and state covariance.
        """
        dim = x.size

        # calculate the sigma points and weights
        sigma = self.compute_sigma(x, P)
        w_mean, w_cov = self.compute_weights(x)

        # propagate sigma points
        for i in range(2 * dim + 1):
            sigma[:, i] += self.compute_state(sigma[:, i], u) * dt

        # compute mean (x)
        x = np.sum(w_mean * sigma, axis=1)

        # compute covariance (P)
        # P = np.copy(Q)
        d = sigma - x[:,np.newaxis]
        P = d @ np.diag(w_cov) @ d.T + Q

        return x, P


    def update(self,x: np.ndarray,
               z: np.ndarray,
               P: np.ndarray,
               R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the state.
        Input:
            x : initial state.
            z : observation.
            P : initial state covariance matrix.
            R : observation noise.
        Return:
            updated state and state covariance.
        """
        dim = x.size

        # calculate the sigma points
        sigma = self.compute_sigma(x, P)
        w_mean, w_cov = self.compute_weights(x)

        # compute sigma for observation
        z_sigma = self.calculate_observation(sigma)

        # compute observation mean
        z_mean = np.sum(w_mean * z_sigma, axis=1)

        # compute observation covariance
        # S = np.copy(R)
        dz = z_sigma - z_mean[:, np.newaxis]
        S = dz @ np.diag(w_cov) @ dz.T + R

        # compute cross covariance
        V = np.zeros((dim, z.size))
        dx = sigma - x[:, np.newaxis]
        V += dx @ np.diag(w_cov) @ dz.T

        # update state mean and covariance
        K = V @ np.linalg.inv(S)   # Kalman gain
        x += K @ (z - z_mean)
        P -= K @ S @ K.T

        return x, P

