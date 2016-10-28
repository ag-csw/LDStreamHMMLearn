from ldshmm.util.util_math import Utility
import math

class Variable_Holder():
    """
    This class serves as holder class for various parameters used in evaluation scripts
    """

    min_eta = 8
    min_scale_window = 8
    min_num_trajectories = 1
    heatmap_size = 3
    min_taumeta = 2
    min_timescale_min = 1


    min_timescaledisp = 2
    min_statconc = math.pow(2,-3)
    mid_statconc = 1 # math.pow(2,0)

    min_omega = 1

    mid_taumeta = Utility.get_mid_value(min_taumeta, heatmap_size)
    mid_eta = Utility.get_mid_value(min_eta, heatmap_size)
    mid_scale_window = Utility.get_mid_value(min_scale_window, heatmap_size)
    mid_num_trajectories = Utility.get_mid_value(min_num_trajectories, heatmap_size)
    mid_timescalemin  = Utility.get_mid_value(min_timescale_min, heatmap_size)
    mid_timescaledisp = Utility.get_mid_value(min_timescaledisp, heatmap_size)
    #mid_statconc = Utility.get_mid_value(min_statconc, heatmap_size)
    mid_omega = Utility.get_mid_value(min_omega, heatmap_size)



    max_eta = min_eta * math.pow(2, heatmap_size - 1)
    max_taumeta = min_taumeta * math.pow(2, heatmap_size - 1)
    shift_max = max_eta * max_taumeta
    window_size_max = mid_scale_window * shift_max
    num_estimations_max = 1  # smallest value within the heatmap

    num_trajectories_max = min_num_trajectories * math.pow(2, heatmap_size - 1)
    len_trajectory = int(window_size_max + num_estimations_max * shift_max + 1)
    num_trajectories_len_trajectory_max = num_trajectories_max * len_trajectory



    product_mid_values = mid_eta * mid_scale_window * mid_num_trajectories
    num_estimations_global = 16 * product_mid_values

    product_mid_values_nonstat = mid_eta * mid_scale_window * mid_num_trajectories
    num_estimations_global_nonstat = 16 * product_mid_values_nonstat