"""
This class is for evaluation purposes.
"""
import time

import numpy as np

from bayesian_qhmm import BayesianQHMM


class Evaluation():
    def __init__(self, qHMM_Family, cluster_prior):
        """
        :param HMM_Family:
        :param cluster_prior: ndarray (2)
        """
        self.qHMM_Family = qHMM_Family
        self.cluster_prior = cluster_prior

    def sample_qHMM_family(self, nsamp):
        """
        :param nsamp: int
           number of samples to be returned
        :return: qHMM
        """
        return self.qHMM_Family.sample(nsamp)

    def simulate_data(self, qHMM, num_trajectories):
        """
        sample from this qHMM
        :param qHMM:
            qHMM model for the simulated data
        :param num_trajectories
            number of trajectories to be returned
        :return: [trajectories]
        """
        return qHMM.simulate()

    def evaluate_performance(self, qHMM, nsamples):
        """
        performance evaluation on the qHMM using simulated data
        :return:
        """
        num_trajectories = 1000
        simulated_data = self.simulate_data(qHMM, num_trajectories)

        window_size = 100
        est1, delt1 = self.elapsed_time_learn(simulated_data, self.cluster_prior, nsamples, window_size)

        window_size = 200
        est2, delt2 = self.elapsed_time_learn(simulated_data, self.cluster_prior, nsamples, window_size)

        order = np.log2(delt2 / delt1)

        return delt1, order

    def elapsed_time_learn(self, simulated_data):
        """
        conduct a single time evaluation for learning model parameters
        :param simulated_data:
        :return: elapsed time
        """
        num_states = 4
        lag = 1
        stride = 'effective'
        nsamples = 100
        init_hmsm = None
        stationary = False
        reversible = False
        connectivity = 'largest'
        mincount_connectivity = '1/n'
        separate = None
        observe_nonempty = True
        dt_traj = '1 step'
        conf = 0.95
        store_hidden = False
        show_progress = False
        t0 = time.time()
        estimator = BayesianQHMM(num_states, lag, stride,
                                 self.cluster_prior,
                                 nsamples, init_hmsm, reversible, stationary,
                                 connectivity, mincount_connectivity, separate, observe_nonempty,
                                 dt_traj, conf, store_hidden, show_progress)
        estimate = estimator.estimate()
        t1 = time.time()
        return estimate, t1 - t0

    def evaluate_accuracy(self, qHMM, simulated_data):
        pass

    def aggregate(self, nagg):
        """calc averages and statistics from evaluation"""
        delts = []
        orders = []
        for j in range(nagg):
            qHMM = self.sample_qHMM_family()
            delt, order = self.evaluate_performance(qHMM)
            delts.append(delt)
            orders.append(order)
        return [np.mean(delts), np.mean(orders)]
