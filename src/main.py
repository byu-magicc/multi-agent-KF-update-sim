#!/usr/bin/env python3

import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from plotters import plot_overview, plot_trajectory_error, Trajectory, Covariance
from simulator import Simulation


def run_simulation(args):
    thread_id, compress_results = args
    np.random.seed(thread_id)
    return Simulation().run(compress_results=compress_results)


def main(num_instances: int):
    num_sigma = 2

    # Run simulations in parallel on multiple cores
    compress_results = True if num_instances > 100 else False
    with Pool(processes=min(num_instances, cpu_count())) as pool:
        results = []
        for result in tqdm(
            pool.imap_unordered(run_simulation,
                                [(i, compress_results) for i in range(num_instances)]),
            total=num_instances,
            desc="Simulating"
        ):
            results.append(result)

    num_vehicles = len(Simulation().vehicles)

    # Extract data for plotting
    poses = []
    covariances = []
    mu_hist = {}
    truth_hist = {}
    Sigma_hist = {}
    for result in results:
        mu_hist_array = result[0]
        truth_hist_array = result[1]
        Sigma_hist_array = result[2]
        keyframe_mu_hist_array = result[3]
        keyframe_Sigma_hist_array = result[4]

        for i in range(num_vehicles):
            poses.append(Trajectory(truth_hist_array[i][:2, :], color="r"))
            poses.append(Trajectory(mu_hist_array[i][:2, :], color="b"))

            if num_instances == 1:
                poses.append(Trajectory(keyframe_mu_hist_array[i][:2, :], color="g"))
                covariances.append(Covariance(Sigma_hist_array[i][-1, :2, :2],
                                              mu_hist_array[i][:2, -1].reshape(-1, 1),
                                              color="k"))

                for j in range(keyframe_mu_hist_array[i].shape[1]):
                    keyframe_mu = keyframe_mu_hist_array[i][:2, j].reshape(-1, 1)
                    keyframe_cov = keyframe_Sigma_hist_array[i][j]
                    covariances.append(Covariance(keyframe_cov[:2, :2],
                                                  keyframe_mu[:2].reshape(-1, 1),
                                                  color="g"))
            else:
                if len(covariances) < num_vehicles:
                    covariances.append(Covariance(Sigma_hist_array[i][-1, :2, :2],
                                                  mu_hist_array[i][:2, -1].reshape(-1, 1),
                                                  color="k"))
            if len(mu_hist) < num_vehicles:
                mu_hist[f'Vehicle {i}'] = [mu_hist_array[i]]
                truth_hist[f'Vehicle {i}'] = [truth_hist_array[i]]
                Sigma_hist[f'Vehicle {i}'] = [Sigma_hist_array[i]]
            else:
                mu_hist[f'Vehicle {i}'].append(mu_hist_array[i])
                truth_hist[f'Vehicle {i}'].append(truth_hist_array[i])
                Sigma_hist[f'Vehicle {i}'].append(Sigma_hist_array[i])

    poses[0].name = "Truth"
    poses[1].name = "Estimate"
    covariances[0].name = f"{num_sigma} Sigma Bound"

    if num_instances <= 100:
        plot_overview(poses, covariances, num_sigma=num_sigma)
        plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, num_sigma=num_sigma)
    else:
        print('Only plotting sigma bounds since instances > 100')
        plot_trajectory_error(mu_hist, truth_hist, Sigma_hist, num_sigma=num_sigma, sigma_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-instances', type=int, default=1,
                        help='Number of instances to run simulation')
    args = vars(parser.parse_args())

    if args['num_instances'] < 1:
        raise ValueError("Number of instances must be at least 1")

    main(**args)
