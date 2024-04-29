import numpy as np
from math import cos, sin
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class UKF:
    def __init__(self, P, Q, R, dt):
        self.P = P
        self.Q = Q
        self.R = R
        self.dt = dt
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
        n = size

        sigma_points = np.zeros((2 * n + 1, size))
        sigma_points[0] = x

        l = self.alpha ** 2 * (size + self.kappa) - size      # scaling factor
        try:
            cov = sqrtm((n + l) * P)
        except:
            print("error")
            print(P)

        for i in range(1, n + 1):
            sigma_points[i] = x + cov[i - 1]
            sigma_points[n + i] = x - cov[i - 1]

        return sigma_points.T

    def fix_covariance(self, covariance, jitter=1e-2):
        """
        Fix the covariance matrix to be positive definite.
        Input:
            covariance : The covariance matrix.
            jitter : The jitter value.
        Return:
            Fixed covariance matrix.
        """
        symmetric = np.allclose(covariance, covariance.T)

        try:
            np.linalg.cholesky(covariance)
            positive_definite = True
        except np.linalg.LinAlgError:
            positive_definite = False

        # If positive definite, return
        if symmetric and positive_definite:
            return covariance

        covariance = (covariance + covariance.T) / 2

        eig_values, eig_vectors = np.linalg.eig(covariance)
        eig_values[eig_values < 0] = 0
        eig_values += jitter

        covariance = eig_vectors.dot(np.diag(eig_values)).dot(eig_vectors.T)

        return self.fix_covariance(covariance, jitter=10 * jitter)

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

    def full_state(self, model, x, u):
        """
        Observation model for the full state.
        Input:
            model : The model object.
            x : The current state.
            u : The input.
        Return:
            Propagated state.
        """

        prev_pos, prev_rot, prev_vel = x[0:3], x[3:6], x[6:9]
        imu_w, imu_f = u[0:3], u[3:6]
        dt = 1
        new_rot = model.attitude_update(prev_pos, prev_rot, prev_vel, imu_w, dt, x[9:12])
        new_vel = model.velocity_update(prev_pos, prev_rot, new_rot, prev_vel, imu_f, dt, x[12:15])
        new_pos = model.position_update(prev_pos, prev_vel, new_vel, dt)
        bias = x[9:15]

        return np.concatenate((new_pos, new_rot, new_vel, bias))

    def error_state(self, model, x, u, gps_pos):
        """
        Observation model for the error state.
        Input:
            model : The model object.
            x : The current state.
            u : The input.
            gps_pos : The gps position.
        Return:
            Propagated state.
        """
        prev_pos, prev_rot, prev_vel = x[0:3], x[3:6], x[6:9]
        imu_w, imu_f = u[0:3], u[3:6]
        dt = 1
        bias_g = np.array([1,1,1]) * 0.01
        bias_a = np.array([1,1,1]) * 0.01
        new_rot = model.attitude_update(prev_pos, prev_rot, prev_vel, imu_w, dt, bias_g)
        new_vel = model.velocity_update(prev_pos, prev_rot, new_rot, prev_vel, imu_f, dt, bias_a)
        new_pos = model.position_update(prev_pos, prev_vel, new_vel, dt)
        error = new_pos - gps_pos
        new_pos = new_pos - error

        return np.concatenate((new_pos, new_rot, new_vel, error))

    def calculate_observation(self, x, system):
        """
        Calculate the state observation.
        Input:
            x : The current state.
            system : The system type.
        Returns:
            The observation.
        """
        # observation matrix
        if system == "error_state":
            C = np.zeros((6, 12))
        elif system == "full_state":
            C = np.zeros((6, 15))

        C[0:3, 0:3] = np.eye(3, 3)  # observe position
        C[3:6, 6:9] = np.eye(3, 3)  # observe velocity

        # observations = np.array([x[0], x[1], x[2], x[6], x[7], x[8]])
        return C @ x


    def predict(self, model, x, u, P, Q, dt, gps_pos, system):
        """
        Predict the next state.
        Input:
            model : the model object.
            x : initial state.
            u : imu, gyroscope input.
            P : initial state covariance matrix.
            Q : process noise.
            dt : time step.
            gps_pos : gps position.
            system : error state or full state.
        Return:
            predicted mean and state covariance.
        """
        dim = x.size

        # calculate the sigma points and weights
        sigma = self.compute_sigma(x, P)
        w_mean, w_cov = self.compute_weights(x)

        # propagate sigma points
        if system == "error_state":
            for i in range(2 * dim + 1):
                sigma[:, i] = self.error_state(model, sigma[:, i], u, gps_pos)
        elif system == "full_state":
            for i in range(2 * dim + 1):
                sigma[:, i] = self.full_state(model, sigma[:, i], u)

        # compute mean (x)
        x = np.sum(w_mean * sigma, axis=1)

        # compute covariance (P)
        # P = np.copy(Q)
        d = sigma - x[:,np.newaxis]
        P = d @ np.diag(w_cov) @ d.T + Q

        return x, P


    def update(self, x, z, P, R, system):
        """
        Update the state.
        Input:
            x : initial state.
            z : observation.
            P : initial state covariance matrix.
            R : covariance.
            system : error state or full state.
        Return:
            updated state and state covariance.
        """
        dim = x.size

        # calculate the sigma points
        sigma = self.compute_sigma(x, P)
        w_mean, w_cov = self.compute_weights(x)

        # compute sigma for observation
        z_sigma = self.calculate_observation(sigma, system)

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
        P = self.fix_covariance(P)

        return x, P

