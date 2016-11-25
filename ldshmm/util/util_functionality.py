from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from ldshmm.util.variable_holder import Variable_Holder
import numpy as np

def estimate_via_sliding_windows(data, num_states, initial=False, lag=1):
    """
    Generates a count matrix from a given list of discrete trajectories

    :param data: list of ndarrays - trajectories
    :param num_states: int - number of states
    :param initial: bool (Default=False) - for initial runs, we add 1e-8 to the returned count matrix to
    avoid having row sums of 0
    :return: scipy.sparse.csr_matrix or numpy.ndarray - The countmatrix in scipy compressed sparse row
        or numpy ndarray format
    """

    C = count_matrix_coo2_mult(data, lag=lag, sliding=False, sparse=False, nstates=num_states)
    if initial:
        C += 1e-8

    return C


def create_value_list(first_value, len_list):
    """
    Generates a list of values, so that list = [first_value, first_value^2, first_value^4, ...] with the size of len_list
    This is used to generate the sequence of parameter values for evaluation purposes.

    :param first_value: int - the minimal value of the list
    :param len_list: int - the length of the returned list
    :return: list of ints
    """

    import math
    list = [int(math.pow(2, x) * first_value) for x in range(0, len_list)]
    return list


def create_value_list_floats(first_value, len_list):
    """
    See create_value_list except that it returns a list of floats

    :param first_value: float - the minimal value of the list
    :param len_list: int - the length of the returned list
    :return: list of floats
    """

    import math
    list = [(math.pow(2, x) * first_value) for x in range(0, len_list)]
    return list

def init_time_and_error_arrays(size):
    """
    Initialize average timing and error arrays for naive and bayes

    :param size: int - size of the returned time and error matrices
    :return: ndarray of format (size x size)
    """

    avg_times_naive = np.zeros((size, size))
    avg_errs_naive = np.zeros((size, size))
    avg_times_bayes = np.zeros((size, size))
    avg_errs_bayes = np.zeros((size, size))
    return avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive


def read_simulated_data(filename):
    """
    Method to read simulated data from a file

    :param filename: str - filename to read simulated data from
    :return: dict where
        * the keys correspond to taumeta values that the data was simulated with
        * the values corrspond to the simulated data itself
    """

    simulated_data = {}
    if filename =="mm" or filename == "qmm":
        for taumeta in create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size):
            ndarr = np.loadtxt("simulated_data_" + filename + str(taumeta), delimiter=",")
            simulated_data[taumeta] = ndarr
        return simulated_data
    elif filename=="tdisp":
        for timescaledisp in create_value_list(Variable_Holder.min_timescaledisp, Variable_Holder.heatmap_size):
            ndarr = np.loadtxt("simulated_data_tdisp"+ str(timescaledisp), delimiter=",")
            simulated_data[timescaledisp] = ndarr
        return simulated_data
    else:
        for statconc in Variable_Holder.statconc_values:
            ndarr = np.loadtxt("simulated_data_statconc"+ str(statconc), delimiter=",")
            simulated_data[statconc] = ndarr
        return simulated_data



def simulate_and_store(qmm1_0_0, num_trajs_simulated=Variable_Holder.num_trajs_simulated, len_trajectory = Variable_Holder.len_trajectory_max, taumeta = Variable_Holder.max_taumeta):
    """
    Method to simulate trajectory data and from a given ConvexCombinationQuasiMM qmm1_0_0 and store it into a file

    :param qmm1_0_0: ConvexCombinationQuasiMM (for instance obtained by sampling the QMMFamily1)
    :param filename: filename to store the trajectory data to
    """

    print("Simulating data")
    qmm1_0_0_scaled = qmm1_0_0.eval(taumeta)
    simulation = qmm1_0_0_scaled.simulate(N=len_trajectory, M=num_trajs_simulated)
    simulation_ndarray = np.asarray(simulation)
    return simulation_ndarray


def simulate_and_store_data(qmm1_0_0, filename):
    """
    Method to simulate trajectory data and from a given ConvexCombinationQuasiMM qmm1_0_0 and store it into a file

    :param qmm1_0_0: ConvexCombinationQuasiMM (for instance obtained by sampling the QMMFamily1)
    :param filename: filename to store the trajectory data to
    """

    dict = {}
    print("Simulating data")
    if filename == "mm" or filename== "qmm":
        for taumeta in create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size):
            data = []
            qmm1_0_0_scaled = qmm1_0_0.eval(taumeta)
            for i in range(0, int(Variable_Holder.max_num_trajectories)):
                simulation = (qmm1_0_0_scaled.simulate(Variable_Holder.len_trajectory_max))
                data.append(simulation)
            dict[taumeta] = data
            print("Done with Taumeta " + str(taumeta))
            np.savetxt("simulated_data_" +filename+  str(taumeta), data, delimiter=",")
        return dict

    elif filename =="tdisp":
        simulate_and_store_data_timescaledisp()
    else:
        simulate_and_store_data_statconc()

def simulate_and_store_data_timescaledisp():
    """
    Method to simulate trajectory data and from for different timescaledisp values and store it into a file

    :param filename: filename to store the trajectory data to
    """

    for timescaledisp in create_value_list(Variable_Holder.min_timescaledisp, Variable_Holder.heatmap_size):
        from ldshmm.util.mm_family import MMFamily1
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states, timescaledisp=timescaledisp)
        mmf1_0_0 = mmf1.sample()[0]
        print("Simulating data")
        data = []
        qmm1_0_0_scaled = mmf1_0_0.eval(Variable_Holder.mid_taumeta)
        max_len_trajectory = Variable_Holder.num_trajectories_len_trajectory_max / Variable_Holder.min_num_trajectories
        for i in range(0, int(Variable_Holder.max_num_trajectories)):
             simulation = (qmm1_0_0_scaled.simulate(int(max_len_trajectory)))
             data.append(simulation)
        print("Done with Timescaledisp " + str(timescaledisp))
        np.savetxt("simulated_data_tdisp" + str(timescaledisp), data, delimiter=",")

def simulate_and_store_data_statconc():
    """
    Method to simulate trajectory data and from for different statconc values and store it into a file

    :param filename: filename to store the trajectory data to
    """

    for statconc in Variable_Holder.statconc_values:
        from ldshmm.util.mm_family import MMFamily1
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states, statconc=statconc)
        mmf1_0_0 = mmf1.sample()[0]
        data = []
        qmm1_0_0_scaled = mmf1_0_0.eval(Variable_Holder.mid_taumeta)
        max_len_trajectory = Variable_Holder.num_trajectories_len_trajectory_max / Variable_Holder.min_num_trajectories
        for i in range(0, int(Variable_Holder.max_num_trajectories)):
             simulation = (qmm1_0_0_scaled.simulate(int(max_len_trajectory)))
             data.append(simulation)
        print("Done with statconc " + str(statconc))
        np.savetxt("simulated_data_statconc" + str(statconc), data, delimiter=",")

def print_tm(qmm1_0_0_scaled):
    """
    Method to print the transition matrices of a SpectralMM that is obtained by the input ConvexCombinationNSMM

    qmm1_0_0_scaled evaluated at each of the max_len_trajectory timepoints.
    :param qmm1_0_0_scaled: ConvexCombinationNSMM
    """

    max_len_trajectory = int(Variable_Holder.num_trajectories_len_trajectory_max / Variable_Holder.min_num_trajectories)
    for i in range(0, max_len_trajectory):
        print(qmm1_0_0_scaled.eval(ground_truth_time=i).trans)


def convert_2d_to_list_of_rows(array2d):
    num_rows = len(array2d)
    array1d = array2d.flatten()
    list_of_rows = np.split(array1d, num_rows)
    return list_of_rows

def get_sub_ndarray (ndarray2d, size_sub_ndarray):
    sub_ndarray = ndarray2d[:size_sub_ndarray]
    return sub_ndarray

def reshape_trajs(trajs_2d, num_trajs_for_reshape):
    num_trajectories, len_trajectory = trajs_2d.shape
    num_slices = int(num_trajectories / num_trajs_for_reshape)
    trajs_3d = trajs_2d.reshape((num_slices, num_trajs_for_reshape, len_trajectory))
    return trajs_3d
