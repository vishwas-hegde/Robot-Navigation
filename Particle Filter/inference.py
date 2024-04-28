import numpy as np
import matplotlib.pyplot as plt
import simulation_ukf
import simulation
import json

def compare_time():
    """
    Compare the time taken by the Unscented Kalman Filter and the Particle Filter.
    :return: None
    """
    filename = 'data/studentdata%d.mat'
    Qa = 100
    Qg = 0.1

    time_ukf = []
    time_pf_250 = []
    time_pf_500 = []
    time_pf_750 = []
    time_pf_1000 = []
    time_pf_2000 = []
    time_pf_5000 = []

    for i in range(1, 8):
        file = filename % i
        _, _, _, _, _, execution_time = simulation_ukf.simulate(file)
        time_ukf.append(execution_time)

        _, _, _, _, _, _, execution_time = simulation.simulate(file, 250, "weighted_avg", Qa=Qa, Qg=Qg)
        time_pf_250.append(execution_time)

        _, _, _, _, _, _, execution_time = simulation.simulate(file, 500, "weighted_avg", Qa=Qa, Qg=Qg)
        time_pf_500.append(execution_time)

        _, _, _, _, _, _, execution_time = simulation.simulate(file, 750, "weighted_avg", Qa=Qa, Qg=Qg)
        time_pf_750.append(execution_time)

        _, _, _, _, _, _, execution_time = simulation.simulate(file, 1000, "weighted_avg", Qa=Qa, Qg=Qg)
        time_pf_1000.append(execution_time)

        _, _, _, _, _, _, execution_time = simulation.simulate(file, 2000, "weighted_avg", Qa=Qa, Qg=Qg)
        time_pf_2000.append(execution_time)

        _, _, _, _, _, _, execution_time = simulation.simulate(file, 5000, "weighted_avg", Qa=Qa, Qg=Qg)
        time_pf_5000.append(execution_time)

    # save the data to a file
    with open('time_comparison.json', 'w') as f:
        json.dump({'ukf': time_ukf, 'pf_250': time_pf_250, 'pf_500': time_pf_500, 'pf_750': time_pf_750, 'pf_1000': time_pf_1000, 'pf_2000': time_pf_2000, 'pf_5000': time_pf_5000}, f)

    # Plot the graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 8), time_ukf, label='UKF')
    ax.plot(range(1, 8), time_pf_250, label='PF 250')
    ax.plot(range(1, 8), time_pf_500, label='PF 500')
    ax.plot(range(1, 8), time_pf_750, label='PF 750')
    ax.plot(range(1, 8), time_pf_1000, label='PF 1000')
    ax.plot(range(1, 8), time_pf_2000, label='PF 2000')
    ax.plot(range(1, 8), time_pf_5000, label='PF 5000')
    ax.legend()
    ax.set_xlabel('Data Set')
    ax.set_ylabel('Time')
    ax.set_title('Time Comparison')
    plt.show()

def compare_rmse_pfmethod():
    """
    Compare the RMSE loss between the Particle Filter with different methods.
    :return: RMSE List
    """
    methods = ['weighted_avg', 'average', 'highest_weight']
    filename = 'data/studentdata%d.mat'
    qa = 100
    qg = 0.1

    rmse_pf_1000 = {}
    rmse_pf_2000 = {}
    rmse_pf_5000 = {}
    for i in range(1, 8):
        file = filename % i

        rmse_pf_1000[file] = {}
        rmse_pf_2000[file] = {}
        rmse_pf_5000[file] = {}
        for method in methods:

            estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(file, 1000, method, Qa=qa, Qg=qg)
            _, rmse_val, _ = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
            rmse_pf_1000[file][method] = rmse_val

            estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(file, 2000, method, Qa=qa, Qg=qg)
            _, rmse_val, _ = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
            rmse_pf_2000[file][method] = rmse_val

            estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(file, 5000, method, Qa=qa, Qg=qg)
            _, rmse_val, _ = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
            rmse_pf_5000[file][method] = rmse_val

    rmse = {}

    rmse['1000'] = rmse_pf_1000
    rmse['2000'] = rmse_pf_2000
    rmse['5000'] = rmse_pf_5000

    # save the data to a file
    with open('rmse_pf_method.json', 'w') as f:
        json.dump(rmse, f)

    # print dict
    print(json.dumps(rmse, indent=4))


def rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history):
    """
    Compute the RMSE loss between the ground truth and the estimated data and the filtered data.
    :param estimated_data: Data estimated from camera data
    :param est_times: Times for estimated data
    :param gt: The ground truth data
    :param gt_tstamp: Times for ground truth data
    :param filtered_data: Data filtered by the Particle Filter
    :param particle_history: History of particles
    :return: RMSE
    """
    rmse_est = np.zeros(len(estimated_data[0]))
    rmse_filter = np.zeros(len(filtered_data[0]))
    rmse_pf = np.zeros(len(filtered_data[0]))

    for i in range(len(estimated_data[0])):
        data_idx_gt = np.argmin(np.abs(gt_tstamp - est_times[i]))
        rmse_est[i] = np.sqrt(np.mean((estimated_data[:3, i] - gt[:3, data_idx_gt]) ** 2))
        rmse_filter[i] = np.sqrt(np.mean((filtered_data[:3, i] - gt[:3, data_idx_gt]) ** 2))
        rmse_pf[i] = np.sqrt(np.mean((particle_history[:, :3, i].mean(axis=0) - gt[:3, data_idx_gt]) ** 2))

    return np.mean(rmse_est), np.mean(rmse_filter), np.mean(rmse_pf)

def rmse_calcuate_ukf(estimated_data, est_times, gt, gt_tstamp, filtered_data):
    """
    Compute the RMSE loss between the ground truth and the estimated data and the filtered data.
    :param estimated_data: Data estimated from camera data
    :param est_times: Times for estimated data
    :param gt: The ground truth data
    :param gt_tstamp: Times for ground truth data
    :param filtered_data: Data filtered by the Particle Filter

    :return: RMSE
    """
    rmse_est = np.zeros(len(estimated_data[0]))
    rmse_filter = np.zeros(len(filtered_data[0]))
    for i in range(len(estimated_data[0])):
        data_idx_gt = np.argmin(np.abs(gt_tstamp - est_times[i]))
        rmse_est[i] = np.sqrt(np.mean((estimated_data[:3, i] - gt[:3, data_idx_gt]) ** 2))
        rmse_filter[i] = np.sqrt(np.mean((filtered_data[:3, i] - gt[:3, data_idx_gt]) ** 2))

    return np.mean(rmse_est), np.mean(rmse_filter)

def noise_compare():
    """
    Compare the RMSE loss between the Particle Filter with different observation noise.
    :return: RMSE List
    """
    filename = 'data/studentdata%d.mat'
    Qa = [90, 100, 110, 120, 130]
    Qg = [0.01, 0.1, 1]
    rmse_pf = {}
    for qa in Qa:
        rmse_pf[qa] = {}
        for qg in Qg:
            rmse_pf[qa][qg] = {}
            for i in range(1, 8):
                file = filename % i
                estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(file, 1000, "weighted_avg", Qa=qa, Qg=qg)
                _, rmse_val, _ = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
                rmse_pf[qa][qg][file] = rmse_val

    # save the data to a file
    with open('rmse_pf_noise.json', 'w') as f:
        json.dump(rmse_pf, f)

    # print dict as a table
    print(json.dumps(rmse_pf, indent=4))

def general_gauge():
    """
    Evaluate Particle Filter performance across all particles.
    :return: None
    """
    filename = 'data/studentdata%d.mat'
    qa = 100
    qg = 0.1
    rmse_values = {}

    for i in range(1, 7):
        file = filename % i

        estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(
            file, 2000, "weighted_avg", Qa=qa, Qg=qg)
        _, rmse_val, rmse_avg = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
        rmse_values[file] = rmse_avg

    # save the data to a file
    with open('rmse_pf_general_gauge.json', 'w') as f:
        json.dump(rmse_values, f)

    # print dict as a table
    print(json.dumps(rmse_values, indent=4))

def filter_performance():
    """
    Evaluate Particle Filter performance for different number of particles.
    :return: None
    """
    filename = 'data/studentdata%d.mat'
    qa = 100
    qg = 0.1
    rmse_250 = []
    rmse_500 = []
    rmse_750 = []
    rmse_1000 = []
    rmse_2000 = []
    rmse_5000 = []

    for j in range(1, 8):
        file = filename % j
        for i in [250, 500, 750, 1000, 2000, 5000]:
            estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(
                file, i, "weighted_avg", Qa=qa, Qg=qg)
            _, rmse_val, rmse_avg = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
            if i == 250:
                rmse_250.append(rmse_val)
            elif i == 500:
                rmse_500.append(rmse_val)
            elif i == 750:
                rmse_750.append(rmse_val)
            elif i == 1000:
                rmse_1000.append(rmse_val)
            elif i == 2000:
                rmse_2000.append(rmse_val)
            elif i == 5000:
                rmse_5000.append(rmse_val)

    # save the data to a file
    with open('rmse_pf_performance.json', 'w') as f:
        json.dump({'250': rmse_250, '500': rmse_500, '750': rmse_750, '1000': rmse_1000, '2000': rmse_2000, '5000': rmse_5000}, f)

    # Plot the graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 8), rmse_250, label='PF 250')
    ax.plot(range(1, 8), rmse_500, label='PF 500')
    ax.plot(range(1, 8), rmse_750, label='PF 750')
    ax.plot(range(1, 8), rmse_1000, label='PF 1000')
    ax.plot(range(1, 8), rmse_2000, label='PF 2000')
    ax.plot(range(1, 8), rmse_5000, label='PF 5000')

    ax.legend()
    ax.set_xlabel('Data Set')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Performance')
    plt.show()

def rmse_compare_with_ukf(R):
    """
    Compare the RMSE loss between the Particle Filter and the Unscented Kalman Filter.
    :return: None
    """
    filename = 'data/studentdata%d.mat'
    qa = 100
    qg = 0.1
    rmse_ukf = []
    rmse_250 = []
    rmse_500 = []
    rmse_750 = []
    rmse_1000 = []
    rmse_2000 = []
    rmse_5000 = []

    for j in range(1, 8):
        file = filename % j
        estimated_data, est_times, gt, gt_tstamp, filtered_data, execution_time = simulation_ukf.simulate(file, R=R)
        _, rmse_val = rmse_calcuate_ukf(estimated_data, est_times, gt, gt_tstamp, filtered_data)
        rmse_ukf.append(rmse_val)
        for i in [250, 500, 750, 1000, 2000, 5000]:
            estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history, execution_time = simulation.simulate(
                file, i, "weighted_avg", R=R, Qa=qa, Qg=qg)
            _, rmse_val, rmse_avg = rmse_calcuate(estimated_data, est_times, gt, gt_tstamp, filtered_data, particle_history)
            if i == 250:
                rmse_250.append(rmse_val)
            elif i == 500:
                rmse_500.append(rmse_val)
            elif i == 750:
                rmse_750.append(rmse_val)
            elif i == 1000:
                rmse_1000.append(rmse_val)
            elif i == 2000:
                rmse_2000.append(rmse_val)
            elif i == 5000:
                rmse_5000.append(rmse_val)

    # save the data to a file
    with open('rmse_pf_comparewithukf.json', 'w') as f:
        json.dump({'250': rmse_250, '500': rmse_500, '750': rmse_750, '1000': rmse_1000, '2000': rmse_2000, '5000': rmse_5000}, f)

    # Plot the graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, 8), rmse_ukf, label='UKF')
    ax.plot(range(1, 8), rmse_250, label='PF 250')
    ax.plot(range(1, 8), rmse_500, label='PF 500')
    ax.plot(range(1, 8), rmse_750, label='PF 750')
    ax.plot(range(1, 8), rmse_1000, label='PF 1000')
    ax.plot(range(1, 8), rmse_2000, label='PF 2000')
    ax.plot(range(1, 8), rmse_5000, label='PF 5000')

    ax.legend()
    ax.set_xlabel('Data Set')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison')
    plt.show()

if __name__ == '__main__':
    # compare_time()
    # compare_rmse_pfmethod()
    # noise_compare()
    # general_gauge()
    filter_performance()
    # rmse_compare_with_ukf(0.01)