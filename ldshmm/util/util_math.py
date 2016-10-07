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
        import math
        num_values_mid = int((num_values-1)/2)
        return int(value * math.pow(2, num_values_mid))

    @staticmethod
    def calc_numsteps(lentraj, nwindow, nstep):
        import math
        numsteps = math.floor((lentraj - nwindow -1)/nstep)
        if numsteps < 0:
            raise Exception
        else:
            return numsteps

    @staticmethod
    def calc_numsteps_mid(nwindow_mid, heatmap_size, nstep_mid):
        import math
        numsteps_mid_tmp = math.ceil(nwindow_mid * (math.pow(2, (heatmap_size-1)/2)-1)/nstep_mid)
        factor = math.ceil(math.log2(numsteps_mid_tmp))+1
        numsteps_mid = math.pow(2,factor)
        return numsteps_mid
