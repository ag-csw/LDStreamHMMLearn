"""
This class holds utility functions.
"""


class Utility():
    ###################################
    #   HMM utility functions
    ###################################
    @staticmethod
    def build_hmm():
        pass

        # @staticmethod
        # def simulate_x_trajectories

    ###################################
    #   Evaluation utility functions
    ###################################
    @staticmethod
    def evaluate_performance():
        pass

    @staticmethod
    def evaluate_complexity():
        pass

    @staticmethod
    def evaluate_emission_matrices():
        pass

    @staticmethod
    def calc_slope(x):
        from scipy.stats.mstats import linregress
        y = range(0, len(x))
        slope, intercept, r_value, p_value, std_err = linregress(y, x)
        return slope

    @staticmethod
    def log_value(x):
        import math
        return math.log2(x)

    @staticmethod
    def get_mid_value(value, num_values):
        #ToDo Document - this is mysterious without mentioning its use for heatmap ranges
        import math
        num_values_mid = int((num_values-1)/2)
        return int(value * math.pow(2, num_values_mid))

    @staticmethod
    def calc_num_estimations(len_trajectory, window_size, shift):
        #ToDo Document
        import math
        num_estimations = math.floor((len_trajectory - window_size)/shift)
        if num_estimations < 0:
            raise Exception
        else:
            return num_estimations

    @staticmethod
    def calc_num_estimations_mid(window_size_mid, heatmap_size, shift_mid):
        #ToDo Document
        import math
        num_estimations_mid_tmp = math.ceil(window_size_mid * (math.pow(2, (heatmap_size-1)/2)-1)/shift_mid)
        factor = math.ceil(math.log2(num_estimations_mid_tmp))+1
        num_estimations_mid = math.pow(2,factor)
        return num_estimations_mid

    @staticmethod
    def calc_deciles(values):
        import numpy as np
        deciles = np.percentile(values, np.arange(0, 100, 10))
        return deciles
