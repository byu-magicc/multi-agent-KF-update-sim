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
        TOTAL_TIME = 120
        INITIAL_POSES = [
            np.array([[0], [0], [np.pi / 4]]),
            np.array([[400], [0], [np.pi / 4]]),
            np.array([[0], [400], [np.pi / 4]])
        ]
        VELOCITY = 15.0
        TRAJECTORY_TYPE = TrajectoryType.SINE
        self.MAX_KEYFRAME_TIME = 10.0
        self.GPS_TIME = 60.0
        INITIAL_UNCERTAINTY_STD = 1e-2

        # Create vehicles
        self.vehicles = [
            Vehicle(pose, VELOCITY, INITIAL_UNCERTAINTY_STD, TRAJECTORY_TYPE, TOTAL_TIME)
            for pose in INITIAL_POSES
        ]
        self.active_vehicles = [True] * len(self.vehicles)

        # Create backend
        sigmas = np.full((3, 1), INITIAL_UNCERTAINTY_STD)
        priors = [
            Prior("0", self.vehicles[0]._ekf.mu[:3], sigmas),
            Prior("1", self.vehicles[1]._ekf.mu[:3], sigmas),
            Prior("2", self.vehicles[2]._ekf.mu[:3], sigmas)
        ]
        self.backend = Backend(priors)

    def run(self):
        """
        Runs the entire simulation, from start to finish.

        Returns:
        mu_hist: list of np.array, shape (num_steps, 5)
            List of estimated poses for each vehicle. [x, y, theta, vx, vy]
        truth_hist: list of np.array, shape (num_steps, 5)
            List of true poses for each vehicle. [x, y, theta, vx, xy]
        Sigma_hist: list of np.array, shape (num_steps, 5, 5)
            List of covariances for each vehicle's estimates.
        """

        # Run simulation
        while any(self.active_vehicles):
            for i, vehicle in enumerate(self.vehicles):
                if self.active_vehicles[i]:
                    # Apply simulated keyframe resets
                    if vehicle.get_current_time() % self.MAX_KEYFRAME_TIME == 0 \
                            and vehicle.get_current_time() != 0:
                        keyframe_mu, keyframe_Sigma = vehicle.keyframe_reset()
                        keyframe_Sigma = np.diag(keyframe_Sigma).reshape(-1, 1)

                        self.backend.add_odometry(
                            Odometry(f"{i}", keyframe_mu, np.sqrt(keyframe_Sigma))
                        )

                    # Apply simulated global measurement
                    if vehicle.get_current_time() == self.GPS_TIME:
                        global_meas = vehicle._truth_hist[:3, vehicle._current_step].reshape(-1, 1)
                        global_meas += np.random.normal([0, 0, 0], [0.5, 0.5, 0]).reshape(-1, 1)
                        vehicle.update(global_meas, np.diag([0.5, 0.5, np.inf])**2)

                    vehicle.step()
                    if not vehicle.is_active():
                        self.active_vehicles[i] = False

        mu_hist = []
        truth_hist = []
        Sigma_hist = []
        keyframe_mu_hist = []
        keyframe_Sigma_hist = []
        for i, vehicle in enumerate(self.vehicles):
            mu, truth, Sigma = vehicle.get_history()
            mu_hist.append(mu)
            truth_hist.append(truth)
            Sigma_hist.append(Sigma)
            keyframe_mu, keyframe_Sigma = self.backend.get_full_trajectory(f"{i}")
            keyframe_mu_hist.append(keyframe_mu)
            keyframe_Sigma_hist.append(keyframe_Sigma)

        return mu_hist, truth_hist, Sigma_hist, keyframe_mu_hist, keyframe_Sigma_hist


if __name__ == "__main__":
    from plotters import plot_trajectory_error, plot_overview, Trajectory, Covariance

    np.random.seed(1)

    simulation = Simulation()

    mu_hist_array, truth_hist_array, Sigma_hist_array = simulation.run()

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

        mu_hist[i] = mu_hist_array[i]
        truth_hist[i] = truth_hist_array[i]
        Sigma_hist[i] = Sigma_hist_array[i]

    plot_overview(poses, covariances)
    plot_trajectory_error(mu_hist, truth_hist, Sigma_hist)
