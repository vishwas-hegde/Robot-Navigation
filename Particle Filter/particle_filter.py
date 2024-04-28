import math

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform, randn
from numpy.linalg import norm
from scipy.stats import norm as normal_dist
import observation_model as obs_model

class ParticleFilter:
    def __init__(self, num_particles, cov, map_size, Qa, Qg):
        self.num_particles = num_particles
        self.map_size = map_size
        self.Qa = Qa
        self.Qg = Qg
        self.cov = cov
        self.min_particles = num_particles // 2
        self.particles_list = np.zeros((self.num_particles, 6))

    def init_particles(self):
        """
        Generate Initial particles
        :return: particles
        """
        x_range = (self.map_size[0], self.map_size[1])
        y_range = (self.map_size[2], self.map_size[3])
        z_range = (self.map_size[4], self.map_size[5])

        # Set the range for the yaw, pitch and roll
        yaw_range = (- np.pi/2, np.pi/2)
        pitch_range = (- np.pi/2, np.pi/2)
        roll_range = (- np.pi/2, np.pi/2)

        low = np.array([x_range[0], y_range[0], z_range[0], yaw_range[0], pitch_range[0], roll_range[0]])
        high = np.array([x_range[1], y_range[1], z_range[1], yaw_range[1], pitch_range[1], roll_range[1]])

        particles = np.random.uniform(low=low, high=high, size=(self.num_particles, 6))          # Generate the particles with uniform distribution

        particles = np.concatenate((particles, np.zeros((self.num_particles, 9))), axis=1)  # Add the remaining state

        particles = np.expand_dims(particles, axis=-1)

        return particles

    def init_particles1(self, x, P):
        """
        Generate Initial particles
        :param x: Initial estimated state
        :param P: Initial covariance
        :return: Particles
        """
        particles = np.random.multivariate_normal(x, P, self.num_particles)                # Generate the particles with normal distribution
        # particles = np.concatenate((particles, np.zeros((self.num_particles, 9))), axis=1)  # Add the remaining state
        particles = np.expand_dims(particles, axis=-1)

        return particles
    def compute_state_vectorized(self, x, ua, uw):
        """
        Compute the state.
        Input:
            x : state.
            ua : sampled acceleration data.
            uw : sampled angular velocity data.
        Return:
            xdot : state derivative.
        """
        xdot = np.zeros((self.num_particles, 15, 1))

        phi, theta, psi = x[:, 3], x[:, 4], x[:, 5]

        R = np.zeros((self.num_particles, 3, 3, 1))
        R[:, 0, 0] = np.cos(psi) * np.cos(theta) - np.sin(psi) * np.sin(theta) * np.sin(phi)
        R[:, 0, 1] = -np.cos(phi) * np.sin(psi)
        R[:, 0, 2] = np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)
        R[:, 1, 0] = np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta)
        R[:, 1, 1] = np.cos(psi) * np.cos(phi)
        R[:, 1, 2] = np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)
        R[:, 2, 0] = -np.cos(phi) * np.sin(theta)
        R[:, 2, 1] = np.sin(phi)
        R[:, 2, 2] = np.cos(phi) * np.cos(theta)
        R = R.reshape((self.num_particles, 3, 3))

        G = np.zeros((self.num_particles, 3, 3, 1))
        G[:, 0, 0] = np.cos(theta)
        G[:, 0, 2] = -np.cos(phi) * np.sin(theta)
        G[:, 1, 1] = 1
        G[:, 1, 2] = np.sin(phi)
        G[:, 2, 0] = np.sin(theta)
        G[:, 2, 2] = np.cos(phi) * np.cos(theta)
        G = G.reshape((self.num_particles, 3, 3))

        xdot[:, 0:3] = x[:, 6:9]
        xdot[:, 3:6] = np.linalg.inv(G) @ (uw - x[:, 9:12])
        xdot[:, 6:9] = np.array([0, 0, -9.81]).reshape((3,1)) + R @ (ua - x[:, 12:15])

        return xdot

    def predict(self, particles, u, dt):
        """
        Predict the state.
        Input:
            particles : particles from previous step.
            u : imu data.
            dt : time step.
        Return:
            particles : predicted particles.
        """
        noise = np.zeros((self.num_particles, 6, 1))

        # Sample from the noise
        noise[:, 0:3] = np.random.normal(
            scale=self.Qg, size=(self.num_particles, 3, 1)
        )
        noise[:, 3:6] = np.random.normal(
            scale=self.Qa, size=(self.num_particles, 3, 1)
        )

        uw = np.tile(u[:3].reshape(3, 1), (self.num_particles, 1, 1))
        ua = np.tile(u[3:6].reshape((3, 1)), (self.num_particles, 1, 1))

        # uw = uw + noise[:, :3]
        ua = ua + noise[:, 3:6]

        xdot = self.compute_state_vectorized(particles, ua, uw)

        # add gyro noise to xdot,
        # Adding noise post computation of xdot seems to provider better result for orientation
        xdot[:, 3:6] = xdot[:, 3:6] + noise[:, :3]

        particles = particles + xdot * dt    # New Particles

        return particles

    def update(self, particles, z):
        """
        Update the weights with observations.
        Input:
            particles : particles.
            z : measurement.
        Return:
            weights : weights.
        """
        C = np.zeros((6, 15))         # Observation matrix
        C[0:6, 0:6] = np.identity(6)

        covariance_diagonal = np.diag(self.cov).reshape((1, 6))

        # Observed particles
        z_particles = ((C @ particles).reshape((self.num_particles, 6)) + covariance_diagonal)

        z_particles = z_particles.reshape((self.num_particles, 6))
        z_particles = np.concatenate((z_particles, np.zeros((self.num_particles, 9))), axis=1)  # Add the rest of the state

        weights = self.update_weights(z_particles, z)   # Update the weights

        return weights
    def update_weights(self, z_particles, z):
        """
        Update the weights.
        Input:
            z_particles : observed particles.
            z : measurement.
        Return:
            weights : weights.
        """
        errors = z_particles[:, 0:6] - z[0:6]   # Compute the errors

        weights = np.exp(-0.5 * np.sum(errors ** 2, axis=1))

        try:
            weights = weights / np.sum(weights)  # Normalize the weights
        except:
            print("Error in weights")
            exit()
        return weights

    def resample(self, particles, weights):
        """
        Resample the particles.
        Input:
            particles : particles.
            weights : weights.
        Return:
            particles : resampled particles.
        """

        resampled_particles = np.zeros((self.num_particles, 15, 1))
        resampled_weights = np.zeros((self.num_particles, 1))

        weights = weights / np.sum(weights)
        c = weights[0]  # Set to our first weight
        i = 0

        r = np.random.uniform(0, 1 / self.num_particles)

        for k in range(0, self.num_particles):
            # Calculate the next sample point
            u = r + (k * 1 / self.num_particles)

            while c < u:
                i += 1
                c += weights[i]

            resampled_particles[k] = particles[i]
            # resampled_weights[k] = weights[i]

        # resampled_weights = resampled_weights / np.sum(resampled_weights)

        return resampled_particles
    def compute_singular_pose(self, particles, weights, method):
        """
        Compute the singular pose.
        Input:
            particles : estimated particles.
            weights : estimated weights.
            method : method to compute the pose.
        Return:
            pose : pose of the drone (position and orientation).
        """
        if method == 'weighted_avg':
            pose = np.sum(particles * weights.reshape(self.num_particles, 1, 1), axis=0)

        elif method == 'highest_weight':
            max_weight = np.argmax(weights)
            pose = particles[max_weight]

        elif method == 'average':
            pose = np.mean(particles, axis=0)

        return pose


