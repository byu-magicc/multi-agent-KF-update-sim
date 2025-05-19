import numpy as np

from backend import Backend, Prior, Odometry, Global, Range
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
        self.GLOBAL_STEP = [667, 1333]
        self.RANGE_MEASUREMENTS = np.array([[100, 0, 1],
                                            [200, 0, 1],
                                            [300, 0, 1],
                                            [400, 0, 1],
                                            [500, 0, 1],
                                            [600, 0, 1],
                                            [700, 0, 1],
                                            [800, 0, 1],
                                            [900, 0, 1],
                                            [1000, 0, 1],
                                            [1100, 0, 1],
                                            [1200, 0, 1],
                                            [1300, 0, 1],
                                            [1400, 0, 1],
                                            [1500, 0, 1]], dtype=int)

        # Measurement uncertainty
        INITIAL_UNCERTAINTY_STD = np.array([0.5, 0.5, np.deg2rad(5)]).reshape(-1, 1)
        self.GLOBAL_MEASUREMENT_STD = np.array([0.5, 0.5, 1e9]).reshape(-1, 1)
        self.RANGE_MEASUREMENT_STD = 1.0

        # Create vehicles
        self.vehicles = [
            Vehicle(initial_position, final_position, INITIAL_UNCERTAINTY_STD, TRAJECTORY_TYPE)
            for initial_position, final_position in zip(INITIAL_POSITIONS, FINAL_POSITIONS)
        ]
        self.active_vehicles = [True] * len(self.vehicles)

        # Create backend
        priors = [
            Prior(f"{i}", self.vehicles[i]._ekf.mu[:3], INITIAL_UNCERTAINTY_STD)
            for i in range(len(INITIAL_POSITIONS))
        ]
        self.backend = Backend(priors)

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
        truth_hist: list of np.array, shape (3, num_steps) [[x, y, theta]].T
            List of true poses for each vehicle at each timestep.
        ekf_hist_mu: list of np.array, shape (3, num_steps)
            List of estimated poses for each vehicle from the EKF at each timestep.
        ekf_hist_Sigma: list of np.array, shape (num_steps, 3, 3)
            List of covariances for each vehicle from the EKF at each timestep.
        backend_hist_mu: list of np.array, shape (3, num_steps)
            List of estimated poses for each vehicle from the backend at each timestep.
            None if compute_backend is False.
        backend_hist_Sigma: list of np.array, shape (num_steps, 3, 3)
            List of covariances for each vehicle from the backend at each timestep.
            None if compute_backend is False.
        """

        # Run simulation
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
        hist_indices = np.linspace(0, self.vehicles[0]._odom_data.shape[1],
                                   num_steps_in_results, dtype=int)
        while any(self.active_vehicles):
            for i, vehicle in enumerate(self.vehicles):
                if self.active_vehicles[i]:
                    # Propagate ekf
                    curr_ekf_mu, curr_ekf_Sigma = vehicle.step()

                    # Add same odom measurement to backend
                    odom_meas = vehicle._odom_data[:, vehicle._current_step - 1].reshape(-1, 1)
                    self.backend.add_odometry(Odometry(f"{i}",
                                                       odom_meas,
                                                       self.vehicles[i]._odom_sigmas))

                    # Apply simulated global measurement
                    if vehicle.get_current_step() in self.GLOBAL_STEP:
                        if i == 0:
                            # Vehicle a
                            global_meas = \
                                vehicle._truth_hist[:, vehicle._current_step].reshape(-1, 1).copy()
                            global_meas[:2] += np.random.normal(0, self.GLOBAL_MEASUREMENT_STD[:2])

                            curr_ekf_mu, curr_ekf_Sigma = vehicle.update(
                                global_meas,
                                np.diag(self.GLOBAL_MEASUREMENT_STD.flatten())**2
                            )
                            self.backend.add_global(Global(f"{i}",
                                                           global_meas,
                                                           self.GLOBAL_MEASUREMENT_STD))

                            # Vehicle b
                            Sigma_global = np.diag(self.GLOBAL_MEASUREMENT_STD.flatten())**2
                            t_a_b, Sigma_t = self.backend.get_transformation(f"{0}", f"{1}")
                            self.vehicles[1].shared_update(global_meas, t_a_b, Sigma_global, Sigma_t)

                    # Apply simulated range measurements
                    if vehicle.get_current_step() in self.RANGE_MEASUREMENTS[:, 0]:
                        curr_meas = self.RANGE_MEASUREMENTS[np.where(
                            self.RANGE_MEASUREMENTS[:, 0] == vehicle.get_current_step()
                        )].flatten()

                        if i == curr_meas[1]:
                            vehicle_0 = self.vehicles[curr_meas[1]]
                            vehicle_1 = self.vehicles[curr_meas[2]]
                            vehicle_0_position = \
                                vehicle_0._truth_hist[:2, vehicle_0._current_step].copy()
                            vehicle_1_position = \
                                vehicle_1._truth_hist[:2, vehicle_1._current_step].copy()
                            range_meas = np.linalg.norm(vehicle_0_position - vehicle_1_position)
                            range_meas += np.random.normal(0, self.RANGE_MEASUREMENT_STD)

                            self.backend.add_range(Range(f"{curr_meas[1]}",
                                                         f"{curr_meas[2]}",
                                                         range_meas,
                                                         self.RANGE_MEASUREMENT_STD))

                    # Get hist results
                    if vehicle.get_current_step() in hist_indices:
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

        return hist_indices, truth_hist, ekf_hist_mu, ekf_hist_Sigma, \
            backend_hist_mu, backend_hist_Sigma


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview, Trajectory, Covariance

    simulation = Simulation(3)

    hist_indices, truth_hist_array, ekf_mu_hist_array, ekf_Sigma_hist_array, \
        backend_mu_hist_array, backend_Sigma_hist_array = simulation.run(compute_backend=True)

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
    plot_trajectory_error(hist_indices, truth_hist, ekf_mu_hist, ekf_Sigma_hist, backend_mu_hist,
                          backend_Sigma_hist, plot_backend=True)
