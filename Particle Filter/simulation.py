import observation_model as obs_model
import numpy as np
import scipy.io as sio
from scipy.linalg import block_diag
import covariances
import particle_filter
import plots
import time
import inference

def map_size(estimated_data):
    """
    Returns the size of the map.
    Inputs:
        estimated_data: Data estimated from camera data
    Returns:
        map: The size of the map
    """
    # Get the maximum and minimum values of the estimated data
    max_x = np.max(estimated_data[0])
    min_x = np.min(estimated_data[0])
    max_y = np.max(estimated_data[1])
    min_y = np.min(estimated_data[1])
    max_z = np.max(estimated_data[2])
    min_z = np.min(estimated_data[2])

    map = np.array([max_x, min_x, max_y, min_y, max_z, min_z])

    # Provide a buffer of 0.5 meters
    buffer = 0.5
    map[0] += buffer
    map[1] -= buffer
    map[2] += buffer
    map[3] -= buffer
    map[4] += buffer
    map[5] -= buffer

    return map

def simulate(filename: str, particle_count = None ,method = "weighted_avg",R = None, Qa = None, Qg = None):
    """
    Simulate the Unscented Kalman Filter.
    Inputs:
        filename: The file containing the data
        particle_count: Number of particles to use
        method: Method to use for estimating the pose
        R : Covariance matrix
        Qa: Acceleration noise
        Qg: Gyroscope noise
    Returns:
        estimated_data: Data estimated from camera data
        est_times: Times for estimated data
        gt: The ground truth data
        gt_tstamp: Times for ground truth data
        filtered_data: Data filtered by the Kalman Filter
        particle_history: History of particles
        execution_time: Time taken for the simulation
    """
    model = obs_model.ObsModel()
    data_list, gt, gt_tstamp = model.process_data(filename)

    positions = []
    orientations = []
    est_times = []
    data = data_list
    for idx, dat in enumerate(data):
        if isinstance(dat['id'], int):
            pos, rot = model.estimate_pose([dat['id']], idx)
            positions.append(pos)
            orientations.append(rot)
            est_times.append(float(dat['t']))
            continue
        if dat['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(dat['id'], idx)
            positions.append(pos)
            orientations.append(rot)
            est_times.append(float(dat['t']))

    estimated_data = np.zeros((6, len(positions)))
    estimated_data[0] = np.array([pos[0] for pos in positions])
    estimated_data[1] = np.array([pos[1] for pos in positions])
    estimated_data[2] = np.array([pos[2] for pos in positions])
    estimated_data[3] = np.array([rot[0] for rot in orientations])
    estimated_data[4] = np.array([rot[1] for rot in orientations])
    estimated_data[5] = np.array([rot[2] for rot in orientations])

    if R is None:
        covariance = covariances.cov_compute(estimated_data, est_times, gt, gt_tstamp)

        R = covariance    # observation noise covariance matrix

    num_particles = particle_count
    P_pos = 0.05 ** 2 * np.eye(3)
    P_rot = 0.05 ** 2 * np.eye(3)
    P_vel = 0.05 ** 2 * np.eye(3)
    P_bias = 0.05 * np.eye(6)
    P = block_diag(P_pos, P_rot, P_vel, P_bias)  # initial state covariance matrix

    map = map_size(estimated_data)
    pf = particle_filter.ParticleFilter(num_particles, R, map, Qa, Qg)

    filtered_data = np.zeros((6, len(estimated_data[0])))

    t = 0   # initial time
    x = gt[:, :][:, 0]  # initial state
    i = 0   # index for filtered data

    particles = pf.init_particles()
    x = np.concatenate((x, np.zeros(3)))
    # particles = pf.init_particles1(x, P)  # Second method to initialize particles

    particle_history = np.zeros((num_particles, 6, len(estimated_data[0])))

    # Compute the time taken for the simulation
    start = time.time()
    for idx, dat in enumerate(data_list):
        pos = None
        rot = None
        if isinstance(dat['id'], int):
            pos, rot = model.estimate_pose([dat['id']], idx)

        elif dat['id'].shape[0] > 1:
            pos, rot = model.estimate_pose(dat['id'], idx)

        if pos is not None:
            dt = dat["t"] - t

            u = np.concatenate((dat['omg'], dat['acc']), axis=0)     # control input

            z = np.array([pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])   # Measurement

            particles = pf.predict(particles, u, dt)   # Prediction step

            weights = pf.update(particles, z)          # Update step

            estimate = pf.compute_singular_pose(particles, weights, method)    # Compute the estimated pose

            particle_history[:, :, i] = particles[:, :6, :].reshape((num_particles, 6,))   # Store the particles

            particles = pf.resample(particles, weights)   # Resample the particles

            t = dat["t"]

            filtered_data[:, i] = estimate[0:6].reshape((6,))   # Store the estimated pose
            i = i + 1

    end = time.time()
    execution_time = end - start  # Time taken for the simulation

    return estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time

if __name__ == "__main__":
    estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulate(r"D:\Spring 2024\Adv Nav\Codes\Nonlinear Kalman Filter\data\studentdata1.mat", 1000, Qa=100, Qg=0.1)
    plots.plot_results(estimated_data, est_times, gt, gt_tstamp, filtered_data)