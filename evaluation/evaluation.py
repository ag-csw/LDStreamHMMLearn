"""
This class is for evaluation purposes.
"""

class Evaluation():

    def __init__(self, HMM_Family, cluster_prior):
        """
        :param HMM_Family:
        :param cluster_prior: ndarray (2)
        """

    def sample_HMM_family(self):
        """
        :return: qHMM
        """
        pass

    def simulate_data(self, qHMM):
        """
        sample from this qHMM
        :param qHMM:
        :return: [trajectories]
        """
        pass

    def evaluate_performance(self, qHMM, simulated_data):
        """
        performance evaluation on the qHMM and simulated data
        :return:
        """
        pass


    def evaluate_accurracy(self, qHMM, simulated_data):
        pass


    def aggregate(self):
        """calc averages and statistics from evaluation"""
        pass