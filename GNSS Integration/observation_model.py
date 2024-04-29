import numpy as np
from scipy.spatial.transform import Rotation as R
import earth


class Observation:
    def __init__(self, filename):
        self.filename = filename
        self.RATE = 7.292115e-5
        self.prev_omega_en = np.array([0, 0, 0])
        self.prev_omega_ie = np.array([0, 0, 0])
        self.bias_g = np.array([1, 1, 1]) * 0.01
        self.bias_a = np.array([1, 1, 1]) * 0.01

    def read_data(self):
        """
        Read the data from the csv file
        :return: data
        """
        data = np.genfromtxt(self.filename, delimiter=',', skip_header=1)

        return data

    def attitude_update(self, prev_pos, prev_rot, prev_vel, imu_w, dt, bias_g):
        """
        Update the attitude using the angular velocity
        Input:
            prev_pos: previous position
            prev_rot: previous rotation
            prev_vel: previous velocity
            imu_w: angular velocity from the IMU
            dt: time step
            bias_g: gyroscope bias
        Return:
            new_rot: updated rotation/attitude
        """
        lat, lon, alt = prev_pos
        rn, re, rp = earth.principal_radii(lat, alt)     # Principal radii
        vn, ve, vd = prev_vel

        imu_w = imu_w - bias_g   # Correct the gyroscope bias

        omega_ie = np.array([[0, -self.RATE, 0], [self.RATE, 0, 0], [0, 0, 0]])

        w_en = np.array([(ve/(re + alt)), (-vn/(rn + alt)), (-ve*np.tan(np.deg2rad(lat))/(re + alt))])

        omega_en = np.array([[0, w_en[2], -w_en[1]], [-w_en[2], 0, w_en[0]], [w_en[1], -w_en[0], 0]])

        omega_ib = np.array([[0, -imu_w[2], imu_w[1]], [imu_w[2], 0, -imu_w[0]], [-imu_w[1], imu_w[0], 0]])

        # Convert to rotation matrix
        prev_rot = R.from_euler('xyz', prev_rot, degrees=True)
        prev_rot = prev_rot.as_matrix()

        # Update the rotation matrix
        new_rot = prev_rot * (np.eye(3) + omega_ib * dt) - (omega_ie + omega_en) * prev_rot * dt

        self.prev_omega_en = omega_en
        self.prev_omega_ie = omega_ie

        # Convert to euler angles
        new_rot = R.from_matrix(new_rot)
        new_rot = new_rot.as_euler('xyz', degrees=True)

        return new_rot

    def velocity_update(self, prev_pos, prev_rot, new_rot, prev_vel, imu_f, dt, bias_a):
        """
        Update the velocity
        Input:
            prev_pos: previous position
            prev_rot: previous rotation
            new_rot: updated rotation
            prev_vel: previous velocity
            imu_f: acceleration from the IMU
            dt: time step
            bias_a: accelerometer bias
        Return:
            new_vel: updated velocity
        """
        lat, lon, alt = prev_pos

        imu_f = imu_f - bias_a   # Correct the accelerometer bias

        # Convert to rotation matrix
        prev_rot = R.from_euler('xyz', prev_rot, degrees=True).as_matrix()
        new_rot = R.from_euler('xyz', new_rot, degrees=True).as_matrix()

        f_n = 1/2 * (prev_rot + new_rot) @ imu_f     # Specific force in the navigation frame

        v_n = prev_vel + dt * (f_n + earth.gravity(lat, alt) - (self.prev_omega_en + 2 * self.prev_omega_ie) @ prev_vel)

        return v_n

    def position_update(self, prev_pos, prev_vel, new_vel, dt):
        """
        Update the position using the velocity
        Input:
            prev_pos: previous position
            prev_vel: previous velocity
            new_vel: updated velocity
            dt: time step
        Return:
            new_pos: updated position
        """
        prn, pre, prp = earth.principal_radii(prev_pos[0], prev_pos[2])      # Previous principal radii

        alt = prev_pos[2] - dt/2 * (prev_vel[2] + new_vel[2])

        lat = prev_pos[0] + dt/2 * ((prev_vel[0]/(prn + prev_pos[2])) + (new_vel[0] / (prn + alt)))

        nrn, nre, nrp = earth.principal_radii(lat, alt)                      # New principal radii

        lon = prev_pos[1] + dt/2 * ((prev_vel[1]/((pre + prev_pos[2]) * np.cos(np.deg2rad(prev_pos[0]))))
                                    + (new_vel[1]/((nre + alt) * np.cos(np.deg2rad(lat)))))

        return np.array([lat, lon, alt])


if __name__ == "__main__":
    obs = Observation("trajectory_data.csv")
    data = obs.read_data()
    time = data[:, 0]
    true_pos = data[:, 1:4]
    true_rot = data[:, 4:7]
    imu_w = data[:, 7:10]
    imu_f = data[:, 10:13]
    gps_pos = data[:, 13:16]
    gps_vel = data[:, 16:19]

    prev_pos = true_pos[0]
    prev_rot = true_rot[0]
    prev_vel = gps_vel[0]

    estimated_pos = []
    estimated_rot = []

    for i in range(0, len(time)):
        dt = 1

        new_rot = obs.attitude_update(prev_pos, prev_rot, prev_vel, imu_w[i], dt)

        new_vel = obs.velocity_update(prev_pos, prev_rot, new_rot, prev_vel, imu_f[i], dt)
        new_pos = obs.position_update(prev_pos, prev_vel, new_vel, dt)

        prev_pos = new_pos
        prev_rot = new_rot
        prev_vel = new_vel

        estimated_pos.append(new_pos)
        estimated_rot.append(new_rot)

    estimated_pos = np.array(estimated_pos)
    estimated_rot = np.array(estimated_rot)

    for idx, i in enumerate(estimated_pos):
        print(i)
        # if idx == 100:
        #     break

