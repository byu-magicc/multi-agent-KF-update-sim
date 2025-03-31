import numpy as np

from backend import Backend, Prior, Odometry, Global, Range
from vehicle import Vehicle, TrajectoryType


class Simulation:
    """
    Simulation class. Contains everything occuring in a single simulation, but no plotting or
    multi-threading.
    """
    def __init__(self):
        # Simulation parameters
        INITIAL_POSITIONS = [
            np.array([[0], [0]]),
            np.array([[400], [0]]),
            np.array([[0], [400]])
        ]
        FINAL_POSITIONS = [
            position.copy() + 1000
            for position in INITIAL_POSITIONS
        ]
        TRAJECTORY_TYPE = TrajectoryType.SINE
        self.GPS_STEP = 750
        INITIAL_UNCERTAINTY_STD = np.array([0.5, 0.5, np.deg2rad(5)]).reshape(-1, 1)
        self.GLOBAL_MEASUREMENT_STD = np.array([0.5, 0.5, np.deg2rad(15)]).reshape(-1, 1)

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

    def run(self, compress_results=False):
        """
        Runs the entire simulation, from start to finish.

        Returns:
        mu_hist: list of np.array, shape (num_steps, 3)
            List of estimated poses for each vehicle. [x, y, theta]
        truth_hist: list of np.array, shape (num_steps, 3)
            List of true poses for each vehicle. [x, y, theta]
        Sigma_hist: list of np.array, shape (num_steps, 3, 3)
            List of covariances for each vehicle's estimates.
        compress_results: bool
            If True, history will be compressed to 100 states.
        """

        # Run simulation
        while any(self.active_vehicles):
            for i, vehicle in enumerate(self.vehicles):
                if self.active_vehicles[i]:
                    # Propagate ekf
                    vehicle.step()

                    # Add same odom measurement to backend
                    odom_meas = vehicle._odom_data[:, vehicle._current_step - 1].reshape(-1, 1)
                    self.backend.add_odometry(Odometry(f"{i}",
                                                       odom_meas,
                                                       self.vehicles[i]._odom_sigmas))

                    # Apply simulated global measurement
                    if vehicle.get_current_step() == self.GPS_STEP:
                        global_meas = \
                            vehicle._truth_hist[:, vehicle._current_step].reshape(-1, 1).copy()
                        global_meas += np.random.normal(0, self.GLOBAL_MEASUREMENT_STD)
                        vehicle.update(global_meas,
                                       np.diag(self.GLOBAL_MEASUREMENT_STD.flatten())**2)

                    if not vehicle.is_active():
                        self.active_vehicles[i] = False

        mu_hist = []
        truth_hist = []
        Sigma_hist = []
        backend_mu_hist = []
        backend_Sigma_hist = []
        for i, vehicle in enumerate(self.vehicles):
            mu, truth, Sigma = vehicle.get_history()
            mu_hist.append(mu)
            truth_hist.append(truth)
            Sigma_hist.append(Sigma)
            backend_mu, backend_Sigma = self.backend.get_full_trajectory(f"{i}")
            backend_mu_hist.append(backend_mu)
            backend_Sigma_hist.append(backend_Sigma)

        if compress_results:
            inx = np.linspace(0, mu_hist[0].shape[1] - 1, 100, dtype=int)
            mu_hist = [mu[:, inx] for mu in mu_hist]
            truth_hist = [truth[:, inx] for truth in truth_hist]
            Sigma_hist = [Sigma[inx] for Sigma in Sigma_hist]
            backend_mu_hist = [mu[:, inx] for mu in backend_mu_hist]
            backend_Sigma_hist = [Sigma[inx] for Sigma in backend_Sigma_hist]

        return mu_hist, truth_hist, Sigma_hist, backend_mu_hist, backend_Sigma_hist


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview, Trajectory, Covariance

    simulation = Simulation()

    mu_hist_array, truth_hist_array, Sigma_hist_array, backend_mu_hist, backend_Sigma_hist \
        = simulation.run()

    poses = []
    covariances = []

    mu_hist = {}
    truth_hist = {}
    Sigma_hist = {}
    for i in range(len(simulation.vehicles)):
        poses.append(Trajectory(mu_hist_array[i][:2, :], name=f"{i} Estimate", color="b"))
        poses.append(Trajectory(truth_hist_array[i][:2, :], name=f"{i} Truth", color="r"))
        covariances.append(Covariance(Sigma_hist_array[i][-1, :2, :2],
                           mu_hist_array[i][:2, -1].reshape(-1, 1),
                           color="b"))

        mu_hist[i] = [mu_hist_array[i]]
        truth_hist[i] = [truth_hist_array[i]]
        Sigma_hist[i] = [Sigma_hist_array[i]]

    plot_overview(poses, covariances)
    plot_trajectory_error(mu_hist, truth_hist, Sigma_hist)
