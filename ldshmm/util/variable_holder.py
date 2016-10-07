from ldshmm.util.util_math import Utility

class Variable_Holder():

    min_eta = 8
    min_scale_win = 8
    min_num_traj = 1
    heatmap_size = 3
    min_taumeta = 2
    min_timescale_min = 1
    min_timescaledisp = 1
    min_statconc = 1/64
    min_omega = 1

    mid_taumeta = Utility.get_mid_value(min_taumeta, heatmap_size)
    mid_eta = Utility.get_mid_value(min_eta, heatmap_size)
    mid_scale_win = Utility.get_mid_value(min_scale_win, heatmap_size)
    mid_num_traj = Utility.get_mid_value(min_num_traj, heatmap_size)
    mid_timescalemin  = Utility.get_mid_value(min_timescale_min, heatmap_size)
    mid_timescaledisp = Utility.get_mid_value(min_timescaledisp, heatmap_size)
    mid_statconc = 1
    mid_omega = Utility.get_mid_value(min_omega, heatmap_size)

    product_mid_values = mid_eta * mid_scale_win * mid_num_traj
    numsteps_global = 16 * product_mid_values


    product_mid_values_nonstat = mid_eta * mid_scale_win * mid_num_traj
    numsteps_global_nonstat = 16 * product_mid_values_nonstat