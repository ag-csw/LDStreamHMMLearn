from ldshmm.util.util_math import Utility

class Variable_Holder():

    min_eta = 8
    min_scale_win = 8
    min_num_traj = 1
    heatmap_size = 3
    min_taumeta = 2

    mid_eta = Utility.get_mid_value(min_eta, heatmap_size)
    mid_scale_win = Utility.get_mid_value(min_scale_win, heatmap_size)
    mid_num_traj = Utility.get_mid_value(min_num_traj, heatmap_size)

    product_mid_values = mid_eta * mid_scale_win * mid_num_traj
    numsteps_global = 16 * product_mid_values