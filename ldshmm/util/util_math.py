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
        return math.log(x)