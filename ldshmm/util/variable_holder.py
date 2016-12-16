from ldshmm.util.util_math import Utility
import math

class Variable_Holder():
    """
    This class serves as holder class for various parameters used in evaluation scripts

    ToDo Document
    There are some justifications for setting some of these constants, and others are
    fairly arbitrary and may be in need of investigation to find the optimum setting.
    """

    min_eta = 8 # minimum value of eta (= ratio shift/taumeta) in the eta heatmap
    min_scale_window = 8 # minimum value of scale_window (= ratio window_size/shift) in the scale_window heatmap 
    min_num_trajectories = 2 # minimum number of trajectories in the num_traj heatmap
    heatmap_size = 1 # size of heatmap grid
    heatmap_factor =  math.pow(2, heatmap_size - 1) # scaling factor over heatmap
    min_taumeta = 4 # minimum value of taumeta in all heatmaps
    min_timescale_min = 1 # ToDo Document

    num_states = 4 # number of observed states in the Markov Model
    numsims = 256#2


    min_timescaledisp = 2 # ToDo Document
    statconc_step = 3 # ToDo Document
    min_statconc = math.pow(2, -statconc_step) # ToDo Document
    mid_statconc = 1 # math.pow(2,0) ToDo Document
    statconc_values = [min_statconc, mid_statconc, math.pow(2, statconc_step)] # ToDo Document
    min_omega = 1 # ToDo Document

    mid_taumeta = Utility.get_mid_value(min_taumeta, heatmap_size) # middle value of taumeta in each heatmap
    mid_eta = Utility.get_mid_value(min_eta, heatmap_size) # middle value of eta, used for constant value of eta in other heatmaps
    mid_scale_window = Utility.get_mid_value(min_scale_window, heatmap_size) # middle value of scale_window, used in other heatmaps
    mid_num_trajectories = Utility.get_mid_value(min_num_trajectories, heatmap_size) # middle number of trajectories, used in other heatmaps
    mid_timescalemin  = Utility.get_mid_value(min_timescale_min, heatmap_size) # ToDo Document
    mid_timescaledisp = Utility.get_mid_value(min_timescaledisp, heatmap_size) # ToDo Document
    #mid_statconc = Utility.get_mid_value(min_statconc, heatmap_size)
    mid_omega = Utility.get_mid_value(min_omega, heatmap_size) # ToDo Document



    max_eta = min_eta * heatmap_factor # maximum value of eta in eta heatmap
    max_taumeta = min_taumeta * heatmap_factor # maximum value of taumeta in all heatmaps #mid_taumeta
    shift_max = max_eta * max_taumeta # the size of the new data window in the bayes method
    window_size_max = mid_scale_window * shift_max # the size of the window in the intialization of the bayes method
    num_estimations_min = 128 #1 # smallest number of Bayes estimations of the transition matrix within the eta and num_traj heatmaps

    max_num_trajectories = 1#min_num_trajectories * heatmap_factor # maximum number of trajectories in num_traj heatmap
    #num_trajectories_len_trajectory_max = min_num_trajectories * len_trajectory_max

    num_trajs_simulated = int(numsims * max_num_trajectories)

    num_transitions_max = int((mid_scale_window + num_estimations_min) * shift_max) # maximum number of transitions needed
    num_trajectories_num_transitions_max = min_num_trajectories * num_transitions_max # constant number of transitions processed in each estimation
    len_trajectory_max = num_transitions_max + 1 # length of trajectory needed from simulations


    product_mid_values = mid_eta * mid_scale_window * mid_num_trajectories # ToDo Document
    num_estimations_global = 16 * product_mid_values # ToDo Document

    product_mid_values_nonstat = mid_eta * mid_scale_window * mid_num_trajectories # ToDo Document
    num_estimations_global_nonstat = 16 * product_mid_values_nonstat # ToDo Document

