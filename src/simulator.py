import numpy as np

from backend import Backend, Prior, Global, Range, IMU
from measurements import get_pseudo_measurement
from vehicle import Vehicle, TrajectoryType


class Simulation:
    """
    Simulation class. Contains everything occuring in a single simulation, but no plotting or
    multi-threading.
    """
    def __init__(self, trajectory_preset):
        # Trajectory type
        if trajectory_preset in [0, 2, 4]:
            INITIAL_POSITIONS = [
                np.array([[350], [0]]),
                np.array([[0], [350]]),
            ]
            FINAL_POSITIONS = [
                position.copy() + np.array([[1414], [1414]])
                for position in INITIAL_POSITIONS
            ]
        elif trajectory_preset in [1, 3, 5]:
            INITIAL_POSITIONS = [
                np.array([[350], [0]]),
                np.array([[0], [350]]),
            ]
            FINAL_POSITIONS = [
                np.array([[2350], [0]]),
                np.array([[0], [2350]]),
            ]
        else:
            raise ValueError(f'Invalid trajectory {trajectory_preset}. Valid trajectories are 0-5.')

        if trajectory_preset in [0, 1]:
            TRAJECTORY_TYPE = TrajectoryType.SINE
        elif trajectory_preset in [2, 3]:
            TRAJECTORY_TYPE = TrajectoryType.LINE
        else:
            TRAJECTORY_TYPE = TrajectoryType.ARC

        # Measurement intervals
        self.GLOBAL_STEP = [50, 100]
        self.RANGE_MEASUREMENTS = np.array([
            [25, 0, 1],
            [50, 0, 1],
            [75, 0, 1],
            [100, 0, 1],
            [125, 0, 1],
            [150, 0, 1],
        ], dtype=int)

        # Measurement uncertainty
        INITIAL_UNCERTAINTY_STD = np.array([5, 5, np.deg2rad(10), 0.5, 0.5]).reshape(-1, 1)
        self.GLOBAL_MEASUREMENT_STD = np.array([0.5, 0.5, 1e9, 1e9, 1e9]).reshape(-1, 1)
        self.RANGE_MEASUREMENT_STD = 1.0
        self.PSEUDO_MEAS_IDX = [0, 1, 2, 3, 4]  # State values to include in psuedo measurement

        # Create vehicles
        self.vehicles = [
            Vehicle(initial_position, final_position, INITIAL_UNCERTAINTY_STD, TRAJECTORY_TYPE, 15)
            for initial_position, final_position in zip(INITIAL_POSITIONS, FINAL_POSITIONS)
        ]
        self.active_vehicles = [True] * len(self.vehicles)

        # Create backend
        priors = [
            Prior(f"{i}", self.vehicles[i]._ekf.mu, INITIAL_UNCERTAINTY_STD)
            for i in range(len(INITIAL_POSITIONS))
        ]
        self.backend = Backend(priors, self.vehicles[0]._IMU_SIGMAS, self.vehicles[0]._DT)

    def run(self, num_steps_in_results=100, compute_backend=False):
        """
        Runs the entire simulation, from start to finish.

        Params:
        num_steps_in_results: int
            Number of timesteps to include in results.
        compute_backend: bool
            If True, the backend solution is computed at every timestep instead of only when
            needed.

        Returns:
        truth_hist: list of np.array, shape (5, num_steps) [[x, y, theta]].T
            List of true poses for each vehicle at each timestep.
        ekf_hist_mu: list of np.array, shape (5, num_steps)
            List of estimated poses for each vehicle from the EKF at each timestep.
        ekf_hist_Sigma: list of np.array, shape (num_steps, 5, 5)
            List of covariances for each vehicle from the EKF at each timestep.
        backend_hist_mu: list of np.array, shape (3, num_steps)
            List of estimated poses for each vehicle from the backend at each timestep.
            None if compute_backend is False.
        backend_hist_Sigma: list of np.array, shape (num_steps, 3, 3)
            List of covariances for each vehicle from the backend at each timestep.
            None if compute_backend is False.
        """

        # Run simulation
        hist_indices = np.linspace(0, self.vehicles[0]._imu_data.shape[1],
                                   num_steps_in_results, dtype=int)
        time_hist = self.vehicles[0]._time_hist[hist_indices]
        truth_hist = [[self.vehicles[i]._truth_hist[:, 0].reshape(-1, 1)]
                      for i in range(len(self.vehicles))]
        ekf_hist_mu = [[self.vehicles[i]._ekf.mu.copy()]
                       for i in range(len(self.vehicles))]
        ekf_hist_Sigma = [[self.vehicles[i]._ekf.Sigma.copy()]
                          for i in range(len(self.vehicles))]
        if compute_backend:
            backend_hist_mu = [[self.backend.get_vehicle_info(f"{i}")[0].copy()]
                               for i in range(len(self.vehicles))]
            backend_hist_Sigma = [[self.backend.get_vehicle_info(f"{i}")[1].copy()]
                                  for i in range(len(self.vehicles))]
        else:
            backend_hist_mu = None
            backend_hist_Sigma = None
        while any(self.active_vehicles):
            for i, vehicle in enumerate(self.vehicles):
                if self.active_vehicles[i]:
                    # Apply IMU
                    imu_meas = vehicle._imu_data[:, vehicle._current_step].reshape(-1, 1)
                    self.backend.add_imu(IMU(f"{i}", imu_meas))
                    curr_ekf_mu, curr_ekf_Sigma = vehicle.step()

                    # Apply simulated global measurement
                    if vehicle.get_current_time() in self.GLOBAL_STEP:
                        if i == 0:
                            # Generate measurement
                            global_meas = \
                                vehicle._truth_hist[:, vehicle._current_step].reshape(-1, 1).copy()
                            global_meas[:2] += np.random.normal(0, self.GLOBAL_MEASUREMENT_STD[:2])

                            # Vehicle a
                            curr_ekf_mu, curr_ekf_Sigma = vehicle.global_update(
                                global_meas,
                                np.diag(self.GLOBAL_MEASUREMENT_STD.flatten())**2
                            )

                            # Vehicle b
                            pre_mu, pre_Sigma = self.backend.get_vehicle_info(f"{1}")
                            self.backend.add_global(Global(f"{0}",
                                                           global_meas,
                                                           self.GLOBAL_MEASUREMENT_STD))
                            post_mu, post_Sigma = self.backend.get_vehicle_info(f"{1}")

                            z, Sigma_z = get_pseudo_measurement(
                                pre_mu[self.PSEUDO_MEAS_IDX],
                                post_mu[self.PSEUDO_MEAS_IDX],
                                pre_Sigma[np.ix_(
                                    self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                                )],
                                post_Sigma[np.ix_(
                                    self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                                )]
                            )
                            z_temp = np.zeros((5, 1))
                            z_temp[self.PSEUDO_MEAS_IDX] = z
                            z = z_temp
                            Sigma_temp = np.eye(5) * 1e9
                            Sigma_temp[np.ix_(
                                self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                            )] = Sigma_z
                            Sigma_z = Sigma_temp
                            self.vehicles[1].global_update(z, Sigma_z)

                    # Apply simulated range measurements
                    if vehicle.get_current_time() in self.RANGE_MEASUREMENTS[:, 0]:
                        curr_meas = self.RANGE_MEASUREMENTS[np.where(
                            self.RANGE_MEASUREMENTS[:, 0] == vehicle.get_current_time()
                        )].flatten()

                        if i == curr_meas[1]:
                            # Generate measurement
                            vehicle_0 = self.vehicles[curr_meas[1]]
                            vehicle_1 = self.vehicles[curr_meas[2]]
                            vehicle_0_position = \
                                vehicle_0._truth_hist[:2, vehicle_0._current_step].copy()
                            vehicle_1_position = \
                                vehicle_1._truth_hist[:2, vehicle_1._current_step].copy()
                            range_meas = np.linalg.norm(vehicle_0_position - vehicle_1_position)
                            range_meas += np.random.normal(0, self.RANGE_MEASUREMENT_STD)

                            # Get pre and post estimates
                            pre_vehicle_0_mu, pre_vehicle_0_Sigma = \
                                self.backend.get_vehicle_info(f"{curr_meas[1]}")
                            pre_vehicle_1_mu, pre_vehicle_1_Sigma = \
                                self.backend.get_vehicle_info(f"{curr_meas[2]}")
                            self.backend.add_range(Range(f"{curr_meas[1]}",
                                                         f"{curr_meas[2]}",
                                                         range_meas,
                                                         self.RANGE_MEASUREMENT_STD))
                            post_vehicle_0_mu, post_vehicle_0_Sigma = \
                                self.backend.get_vehicle_info(f"{curr_meas[1]}")
                            post_vehicle_1_mu, post_vehicle_1_Sigma = \
                                self.backend.get_vehicle_info(f"{curr_meas[2]}")

                            # Calculate and apply psuedo measurements
                            z_0, Sigma_z_0 = get_pseudo_measurement(
                                pre_vehicle_0_mu[self.PSEUDO_MEAS_IDX],
                                post_vehicle_0_mu[self.PSEUDO_MEAS_IDX],
                                pre_vehicle_0_Sigma[np.ix_(
                                    self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                                )],
                                post_vehicle_0_Sigma[np.ix_(
                                    self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                                )]
                            )
                            z_temp = np.zeros((5, 1))
                            z_temp[self.PSEUDO_MEAS_IDX] = z_0
                            z_0 = z_temp
                            Sigma_temp = np.eye(5) * 1e9
                            Sigma_temp[np.ix_(
                                self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                            )] = Sigma_z_0
                            Sigma_z_0 = Sigma_temp

                            z_1, Sigma_z_1 = get_pseudo_measurement(
                                pre_vehicle_1_mu[self.PSEUDO_MEAS_IDX],
                                post_vehicle_1_mu[self.PSEUDO_MEAS_IDX],
                                pre_vehicle_1_Sigma[np.ix_(
                                    self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                                )],
                                post_vehicle_1_Sigma[np.ix_(
                                    self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                                )]
                            )
                            z_temp = np.zeros((5, 1))
                            z_temp[self.PSEUDO_MEAS_IDX] = z_1
                            z_1 = z_temp
                            Sigma_temp = np.eye(5) * 1e9
                            Sigma_temp[np.ix_(
                                self.PSEUDO_MEAS_IDX, self.PSEUDO_MEAS_IDX
                            )] = Sigma_z_1
                            Sigma_z_1 = Sigma_temp

                            vehicle_0.global_update(z_0, Sigma_z_0)
                            vehicle_1.global_update(z_1, Sigma_z_1)

                    # Get hist results
                    if vehicle._current_step in hist_indices:
                        truth_hist[i].append(
                            vehicle._truth_hist[:, vehicle._current_step].reshape(-1, 1)
                        )
                        ekf_hist_mu[i].append(curr_ekf_mu)
                        ekf_hist_Sigma[i].append(curr_ekf_Sigma)

                        if compute_backend:
                            backend_mu, backend_Sigma = self.backend.get_vehicle_info(f"{i}")
                            backend_hist_mu[i].append(backend_mu)
                            backend_hist_Sigma[i].append(backend_Sigma)

                    # Stop simulation if completed
                    if not vehicle.is_active():
                        self.active_vehicles[i] = False

        for i, vehicle in enumerate(self.vehicles):
            truth_hist[i] = np.hstack(truth_hist[i])
            ekf_hist_mu[i] = np.hstack(ekf_hist_mu[i])
            ekf_hist_Sigma[i] = np.array(ekf_hist_Sigma[i])

            if compute_backend:
                backend_hist_mu[i] = np.hstack(backend_hist_mu[i])
                backend_hist_Sigma[i] = np.array(backend_hist_Sigma[i])

        return time_hist, truth_hist, ekf_hist_mu, ekf_hist_Sigma, \
            backend_hist_mu, backend_hist_Sigma


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview, Trajectory, Covariance

    np.random.seed(0)
    simulation = Simulation(0)

    time_hist, truth_hist_array, ekf_mu_hist_array, ekf_Sigma_hist_array, \
        backend_mu_hist_array, backend_Sigma_hist_array = simulation.run(compute_backend=True,
                                                                         num_steps_in_results=100)

    poses = []
    covariances = []

    truth_hist = {}
    ekf_mu_hist = {}
    ekf_Sigma_hist = {}
    backend_mu_hist = {}
    backend_Sigma_hist = {}
    for i in range(len(simulation.vehicles)):
        poses.append(Trajectory(truth_hist_array[i][:2, :], name=f"{i} Truth", color="r"))
        poses.append(Trajectory(ekf_mu_hist_array[i][:2, :], name=f"{i} EKF", color="b"))
        covariances.append(Covariance(ekf_Sigma_hist_array[i][-1, :2, :2],
                           ekf_mu_hist_array[i][:2, -1].reshape(-1, 1),
                           color="b"))
        poses.append(Trajectory(backend_mu_hist_array[i][:2, :], name=f"{i} FG", color="g"))
        covariances.append(Covariance(backend_Sigma_hist_array[i][-1, :2, :2],
                           backend_mu_hist_array[i][:2, -1].reshape(-1, 1),
                           color="g"))

        truth_hist[i] = [truth_hist_array[i]]
        ekf_mu_hist[i] = [ekf_mu_hist_array[i]]
        ekf_Sigma_hist[i] = [ekf_Sigma_hist_array[i]]
        backend_mu_hist[i] = [backend_mu_hist_array[i]]
        backend_Sigma_hist[i] = [backend_Sigma_hist_array[i]]

    plot_overview(poses, covariances)
    plot_trajectory_error(time_hist, truth_hist, ekf_mu_hist, ekf_Sigma_hist, backend_mu_hist,
                          backend_Sigma_hist, plot_backend=True)
