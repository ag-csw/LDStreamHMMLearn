from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from ldshmm.util.variable_holder import Variable_Holder
import numpy as np

def estimate_via_sliding_windows(data, num_states):
    C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates=num_states)
    return C


def create_value_list(first_value, len_list):
    import math
    list = [int(math.pow(2, x) * first_value) for x in range(0, len_list)]
    return list


def create_value_list_floats(first_value, len_list):
    import math
    list = [(math.pow(2, x) * first_value) for x in range(0, len_list)]
    return list

def init_time_and_error_arrays(heatmap_size):
    # initialize average timing and error arrays for naive and bayes
    avg_times_naive = np.zeros((heatmap_size, heatmap_size))
    avg_errs_naive = np.zeros((heatmap_size, heatmap_size))
    avg_times_bayes = np.zeros((heatmap_size, heatmap_size))
    avg_errs_bayes = np.zeros((heatmap_size, heatmap_size))
    return avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive

def init_error_arrays_new(heatmap_size):
    # initialize average error arrays for naive and bayes
    avg_errs_naive = np.zeros((heatmap_size, heatmap_size))
    avg_errs_naive_new = np.zeros((heatmap_size, heatmap_size))
    avg_errs_bayes = np.zeros((heatmap_size, heatmap_size))
    return avg_errs_bayes, avg_errs_naive_new, avg_errs_naive


def read_simulated_data():
    max_len_trajectory = Variable_Holder.num_trajectories_len_trajectory_max / Variable_Holder.min_num_trajectories
    simulated_data = {}
    for taumeta in create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size):
        ndarr = np.loadtxt("simulated_data" + str(taumeta), delimiter=",")
        print(taumeta, ndarr.shape)
        simulated_data[taumeta] = ndarr
    return simulated_data

def simulate_and_store_data(qmm1_0_0):
    for taumeta in create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size):
        data = []
        qmm1_0_0_scaled = qmm1_0_0.eval(taumeta)
        max_len_trajectory = Variable_Holder.num_trajectories_len_trajectory_max / Variable_Holder.min_num_trajectories

        for i in range(0, int(Variable_Holder.max_num_trajectories)):
            print(qmm1_0_0_scaled.eval(ground_truth_time=i).trans)
            simulation = (qmm1_0_0_scaled.simulate(int(max_len_trajectory)))
            data.append(simulation)
        np.savetxt("simulated_data" + str(taumeta), data, delimiter=",")


