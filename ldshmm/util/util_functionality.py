from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
import numpy as np

def estimate_via_sliding_windows(data, num_states):
    C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates=num_states)
    return C


def create_value_list(first_value, len_list):
    import math
    list = [int(math.pow(2, x) * first_value) for x in range(0, len_list)]
    return list


def init_time_and_error_arrays(heatmap_size):
    # initialize average timing and error arrays for naive and bayes
    avg_times_naive = np.zeros((heatmap_size, heatmap_size))
    avg_errs_naive = np.zeros((heatmap_size, heatmap_size))
    avg_times_bayes = np.zeros((heatmap_size, heatmap_size))
    avg_errs_bayes = np.zeros((heatmap_size, heatmap_size))
    return avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive

def init_error_arrays_new(heatmap_size):
    # initialize average timing and error arrays for naive and bayes
    avg_errs_naive = np.zeros((heatmap_size, heatmap_size))
    avg_errs_naive_new = np.zeros((heatmap_size, heatmap_size))
    avg_errs_bayes = np.zeros((heatmap_size, heatmap_size))
    return avg_errs_bayes, avg_errs_naive_new, avg_errs_naive