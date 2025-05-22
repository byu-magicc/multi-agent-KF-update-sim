from functools import partial
from typing import List, Optional

import gtsam
import numpy as np


class Prior:
    """
    Struct for storing priors.
    """
    def __init__(self, vehicle: str, prior_mean: np.ndarray, prior_sigmas: np.ndarray,
                 velocity_mean: np.ndarray, velocity_sigmas: np.ndarray):
        """
        Params:
        vehicle: str
            The name of the vehicle.
        prior_mean: np.ndarray(3,1)
            The mean of the prior pose (x, y, theta).
        prior_sigmas: np.ndarray(3,1)
            The standard deviation of the prior pose (x, y, theta).
        velocity_mean: np.ndarray(2,1)
            The mean of the prior velocity (vx, vy).
        velocity_sigmas: np.ndarray(2,1)
            The standard deviation of the prior velocity (vx, vy).
        """
        assert prior_mean.shape == (3, 1)
        assert prior_sigmas.shape == (3, 1)
        assert velocity_mean.shape == (2, 1)
        assert velocity_sigmas.shape == (2, 1)

        self.vehicle = vehicle

        self.pos = np.zeros(3)
        self.pos[:2] = prior_mean[:2].flatten()

        self.att = np.zeros(3)
        self.att[-1] = prior_mean[-1].flatten()

        self.pos_sigmas = np.ones(3) * 1e-9
        self.pos_sigmas[:2] = prior_sigmas[:2].flatten()

        self.att_sigmas = np.ones(3) * 1e-9
        self.att_sigmas[-1] = prior_sigmas[-1].flatten()

        self.vel = np.zeros(3)
        self.vel[:2] = velocity_mean.flatten()

        self.vel_sigmas = np.ones(3) * 1e-9
        self.vel_sigmas[:2] = velocity_sigmas.flatten()


class IMU:
    """
    Struct for storing IMU measurements.
    """
    def __init__(self, vehicle: str, mean: np.ndarray):
        """
        Params:
        vehicle: str
            The name of the vehicle.
        mean: np.ndarray(3,1)
            The mean of the IMU measurement (ax, ay, omega).
        """
        assert mean.shape == (3, 1)

        self.vehicle = vehicle

        self.accel = np.zeros(3)
        self.accel[:2] = mean[:2].flatten()

        self.gyro = np.zeros(3)
        self.gyro[-1] = mean[-1].flatten()


class Backend:
    """
    Multi-agent backend, utilizing GTSAM for factor graph optimization.
    """
    def __init__(self, priors: List[Prior], imu_sigmas: np.ndarray, imu_dt: float):
        """
        Params:
        priors: List[Prior]
            List of prior for each vehicle. Determines the number of vehicles in the system.
        imu_sigmas: np.ndarray(3,1)
            The standard deviation of the IMU measurements (ax, ay, omega).
        imu_dt: float
            The time step for the IMU measurements.
        """
        assert imu_sigmas.shape == (3, 1)

        # Check for duplicate vehicle names
        vehicles = [prior.vehicle for prior in priors]
        if len(vehicles) != len(set(vehicles)):
            raise ValueError("Duplicate vehicle names in priors.")

        # Create gtsam components
        self.graph = gtsam.NonlinearFactorGraph()
        self.params = gtsam.LevenbergMarquardtParams()

        # Add the priors, keeping track of the vehicle pose ids
        self.next_id = 0
        self.vehicle_pose_ids = {}
        self.vehicle_velocity_ids = {}
        self.vehicle_bias_ids = {}
        self.current_estimates = gtsam.Values()
        for prior in priors:
            p = gtsam.Point3(*prior.pos)
            r = gtsam.Rot3.RzRyRx(*prior.att)

            self.graph.add(
                gtsam.PriorFactorPose3(
                    self.next_id,
                    gtsam.Pose3(r, p),
                    gtsam.noiseModel.Diagonal.Sigmas(np.append(
                        prior.att_sigmas, prior.pos_sigmas
                    ))
                )
            )
            self.current_estimates.insert(self.next_id, gtsam.Pose3(r, p))
            self.vehicle_pose_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

            self.graph.add(
                gtsam.PriorFactorVector(
                    self.next_id,
                    prior.vel,
                    gtsam.noiseModel.Diagonal.Sigmas(prior.vel_sigmas)
                )
            )
            self.current_estimates.insert(self.next_id, prior.vel)
            self.vehicle_velocity_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

            bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
            self.graph.add(
                gtsam.PriorFactorConstantBias(
                    self.next_id,
                    bias,
                    gtsam.noiseModel.Isotropic.Sigma(6, 1e-9)
                )
            )
            self.current_estimates.insert(self.next_id, bias)
            self.vehicle_bias_ids[prior.vehicle] = [self.next_id]
            self.next_id += 1

        # Various initializations
        self.graph_outdated = True
        self.imu_storage = {}
        for prior in priors:
            self.imu_storage[prior.vehicle] = []
        self.imu_dt = imu_dt

        # IMU pre-integration object
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(1e-9)
        accel_Sigma = np.eye(3) * 1e-9
        accel_Sigma[0, 0] = imu_sigmas.item(0)**2
        accel_Sigma[1, 1] = imu_sigmas.item(1)**2
        gyro_Sigma = np.eye(3) * 1e-9
        gyro_Sigma[2, 2] = imu_sigmas.item(2) ** 2
        params.setGyroscopeCovariance(gyro_Sigma)
        params.setAccelerometerCovariance(accel_Sigma)
        params.setIntegrationCovariance(1e-9 * np.eye(3))
        params.setBiasAccCovariance(1e-9 * np.eye(3))
        params.setBiasOmegaCovariance(1e-9 * np.eye(3))
        params.setBiasAccOmegaInit(1e-9 * np.eye(6))
        self.pim = gtsam.PreintegratedCombinedMeasurements(
            params,
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        )

    def get_vehicle_info(self, vehicle: str):
        """
        Optimizes the graph and returns the estimated pose and uncertainty for the specified
        vehicle.

        vehicle: str
            The name of the vehicle.

        Return: (np.ndarray(3,1), np.ndarray(3,3))
            The estimated pose and uncertainty of the vehicle.
        """

        # Check the vehicle name exists
        if vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {vehicle} not found in priors.")

        # Update the graph and return the estimated pose and uncertainty
        self._create_imu_edge(vehicle)
        self._update()
        rotation = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).rotation()
        translation = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).translation()
        x = translation.item(0)
        y = translation.item(1)
        theta = rotation.yaw()
        pose = np.array([x, y, theta]).reshape(-1, 1)

        # Get the covariance of the last pose in global frame
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        covariance = marginals.marginalCovariance(self.vehicle_pose_ids[vehicle][-1])

        R = np.eye(6)
        R[3:, 3:] = rotation.matrix()
        M = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0]])

        cov_2d = M @ R @ covariance @ R.T @ M.T

        return pose, cov_2d

    def get_full_trajectory(self, vehicle: str):
        """
        Optimizes the graph and returns the full trajectory for the specified vehicle.

        vehicle: str
            The name of the vehicle.

        Return: (np.ndarray(3,n), np.ndarray(n,3,3))
            poses and uncertainty, in global (not local) frame
        """

        # Check the vehicle name exists
        if vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {vehicle} not found in priors.")

        # Update the graph and get full trajectory
        self._create_imu_edge(vehicle)
        self._update()
        poses = []
        for i in self.vehicle_pose_ids[vehicle]:
            p = self.current_estimates.atPose3(i).translation()
            r = self.current_estimates.atPose3(i).rotation()
            poses.append(np.array([[p.item(0), p.item(1), r.yaw()]]).T)
        poses = np.hstack(poses)

        # Get the covariances in global frame
        marginals = gtsam.Marginals(self.graph, self.current_estimates)
        covariances = []
        M = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0]])
        for i in self.vehicle_pose_ids[vehicle]:
            cov = marginals.marginalCovariance(i)

            R = np.eye(6)
            R[3:, 3:] = self.current_estimates.atPose3(i).rotation().matrix()
            cov_2d = M @ R @ cov @ R.T @ M.T

            covariances.append(cov_2d)
        covariances = np.array(covariances)

        return poses, covariances

    def add_imu(self, imu: IMU):
        """
        Add an IMU measurement to the graph for the specified vehicle.
        """

        # Check the vehicle name already exists
        if imu.vehicle not in self.vehicle_pose_ids:
            raise ValueError(f"Vehicle name {imu.vehicle} not found in priors.")

        # Add to the IMU storage
        self.imu_storage[imu.vehicle].append(imu)

        self.graph_outdated = True

    def _create_imu_edge(self, vehicle: str):
        """
        Create IMU pre-integration factors for the associated vehicle and add it to the graph.
        """

        # Return if nothing to do
        if self.imu_storage[vehicle] == []:
            return

        # Integrate measurements
        self.pim.resetIntegration()
        for meas in self.imu_storage[vehicle]:
            self.pim.integrateMeasurement(meas.accel, meas.gyro, self.imu_dt)
        self.imu_storage[vehicle] = []

        # Add to FG
        self.graph.push_back(
            gtsam.CombinedImuFactor(
                self.vehicle_pose_ids[vehicle][-1],
                self.vehicle_velocity_ids[vehicle][-1],
                self.next_id,
                self.next_id + 1,
                self.vehicle_bias_ids[vehicle][-1],
                self.next_id + 2,
                self.pim
            )
        )

        # Add estimated pose
        translation = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).translation()
        rotation = self.current_estimates.atPose3(self.vehicle_pose_ids[vehicle][-1]).rotation()
        R = rotation.matrix()
        translation = translation + R @ self.pim.deltaPij()
        rotation = rotation.compose(self.pim.deltaRij())
        self.current_estimates.insert(
            self.next_id,
            gtsam.Pose3(rotation, translation)
        )

        # Add estimated velocity
        velocity = self.current_estimates.atVector(self.vehicle_velocity_ids[vehicle][-1])
        velocity = velocity + R @ self.pim.deltaVij()
        self.current_estimates.insert(
            self.next_id + 1,
            velocity
        )

        # Add zero bias
        self.current_estimates.insert(
            self.next_id + 2,
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        )

        self.vehicle_pose_ids[vehicle].append(self.next_id)
        self.vehicle_velocity_ids[vehicle].append(self.next_id + 1)
        self.vehicle_bias_ids[vehicle].append(self.next_id + 2)
        self.next_id += 3


    def _update(self):
        """
        Update the graph if it is outdated.
        """
        if self.graph_outdated:
            self.current_estimates = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.current_estimates, self.params
            ).optimize()
            self.graph_outdated = False


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse

    prior_pos_noise = np.array([[0.1], [0.2], [0.3]])
    prior_vel_noise = np.array([[0.04], [0.05]])
    imu_noise = np.array([[0.1], [0.1], [0.1]])
    dt = 0.1

    priors = [
        Prior("A", np.array([[0], [0], [0]]), prior_pos_noise,
              np.array([[0], [0]]), prior_vel_noise),
        Prior("B", np.array([[0], [5], [np.pi / 4]]), prior_pos_noise,
              np.array([[0], [0]]), prior_vel_noise),
    ]
    imu_a = IMU("A", np.array([[0.1], [-0.1], [-0.01]]))
    imu_b = IMU("B", np.array([[0.3], [-0.3], [0.01]]))

    backend = Backend(priors, imu_noise, dt)

    for _ in range(100):
        backend.add_imu(imu_a)
        backend.add_imu(imu_b)
    backend._create_imu_edge("A")
    backend._create_imu_edge("B")
    backend._update()
    for _ in range(100):
        backend.add_imu(imu_a)
        backend.add_imu(imu_b)
    backend._create_imu_edge("A")
    backend._create_imu_edge("B")

    poses_1, covariances_1 = backend.get_full_trajectory("A")
    poses_2, covariances_2 = backend.get_full_trajectory("B")

    plt.figure()
    plt.plot(poses_1[0, :], poses_1[1, :], color="b", marker="o", label="A")
    plt.plot(poses_2[0, :], poses_2[1, :], color="g", marker="o", label="B")

    for i in range(poses_1.shape[1]):
        num_sigma = 2
        cov = covariances_1[i][:2, :2]
        mean = poses_1[:2, i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                      width=num_sigma * np.sqrt(eigvals[0]) * 2,
                      height=num_sigma * np.sqrt(eigvals[1]) * 2,
                      angle=np.rad2deg(angle),
                      edgecolor="r",
                      facecolor="none",
                      zorder=4,
                      )
        plt.gca().add_artist(ell)
    for i in range(poses_2.shape[1]):
        num_sigma = 2
        cov = covariances_2[i][:2, :2]
        mean = poses_2[:2, i]
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        ell = Ellipse(xy=(mean[0], mean[1]),
                  width=num_sigma * np.sqrt(eigvals[0]) * 2,
                  height=num_sigma * np.sqrt(eigvals[1]) * 2,
                  angle=np.rad2deg(angle),
                  edgecolor="r",
                  facecolor="none",
                  zorder=4,
                  )
        plt.gca().add_artist(ell)

    plt.axis("equal")
    plt.legend()
    plt.grid()
    plt.show()


