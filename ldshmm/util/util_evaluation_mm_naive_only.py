from time import process_time

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.mm_family import MMFamily1

class Evaluation_Holder_MM():
    """
   Class holding all evaluation functions used for a MMMScaled.
   See util_evaluation.py - The evaluation functions are similar except for the underlying model.
   """

    def __init__(self, mm1_0_0, simulate=True, filename="mm", init_run = False):
        """
        Info: This constructor reads simualted data from a previously created file simulated_data_mm ...

        :param mm1_0_0: MMMScaled (for instance obtained by sampling the MMFamily1)
        :param simulate: bool (default=True) - decide whether data should be simulated in the constructor
        :param init_run: bool (default=True)
        """
        if init_run:
            # ToDo Check which data to use for init run
            estimate_via_sliding_windows(data=[], num_states=Variable_Holder.num_states)
        self.mm1_0_0 = mm1_0_0
        if simulate:
            self.simulated_data = simulate_and_store(mm1_0_0)

    def test_mid_values(self, mm1_0_0=None, simulated_data=None, eta=Variable_Holder.mid_eta, scale_window = Variable_Holder.mid_scale_window):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        # ToDo Document The formulas used here need justification
        self.taumeta = Variable_Holder.mid_taumeta
        self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
        self.shift = eta * self.taumeta
        self.window_size = scale_window * self.shift
        self.num_trajectories = len(self.simulated_data)
        len_trajectory = Variable_Holder.len_trajectory_max
        self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size,
                                                            self.shift)
        self.r = (self.window_size - self.shift) / self.window_size

        self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                len_trajectory, self.num_trajectories, eta,
                                scale_window)

        dataarray = np.asarray(self.simulated_data)
        errors_naive = self.error_naive(dataarray)

        return errors_naive


    """
    ######
    ### Naive Error Calculation Methods (Taumeta)
    #####
    """

    def test_taumeta_eta(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # ToDo Document The formulas used here need justification
                self.taumeta = taumeta
                self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                scale_window = Variable_Holder.mid_scale_window
                len_trajectory = Variable_Holder.len_trajectory_max
                self.window_size = scale_window * self.shift
                self.num_trajectories = len(simulated_data)

                self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size
                # TODO Print out only intermediate variables that are used, not Variable_Holder fields that may or may not be correct
                # TODO calculate the actual trajectory length that is processed from
                actual_len_traj_processed = self.window_size + self.num_estimations*self.shift + 1
                print("Actual processed trajectory length:", actual_len_traj_processed)
                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        len_trajectory, self.num_trajectories, eta, scale_window)

                log_avg_err_naive = self.helper(len_trajectory, self.num_trajectories)

                avg_errs_naive[two][one] = log_avg_err_naive

        return avg_errs_naive, taumeta_values, eta_values

    def test_taumeta_scale_window(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                len_trajectory = Variable_Holder.len_trajectory_max
                self.shift = eta * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = len(simulated_data)
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory_max,
                                                                    self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size
                actual_len_traj_processed = self.window_size + self.num_estimations * self.shift + 1
                print("Actual processed trajectory length:", actual_len_traj_processed)
                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_avg_err_naive = self.helper(len_trajectory, self.num_trajectories)

                avg_errs_naive[two][one] = log_avg_err_naive

        return avg_errs_naive, taumeta_values, scale_window_values

    def test_taumeta_num_traj(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):
                self.taumeta = taumeta
                self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                scale_window = Variable_Holder.mid_scale_window
                self.shift = eta * self.taumeta
                self.window_size = scale_window * self.shift
                self.num_trajectories = num_traj
                len_trajectory = int(
                    Variable_Holder.num_trajectories_num_transitions_max / self.num_trajectories) + 1

                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size
                actual_len_traj_processed = self.window_size + self.num_estimations * self.shift + 1
                print("Actual processed trajectory length:", actual_len_traj_processed)
                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        len_trajectory, self.num_trajectories, eta, scale_window)

                log_avg_err_naive = self.helper(len_trajectory, self.num_trajectories)

                avg_errs_naive[two][one] = log_avg_err_naive

        return avg_errs_naive, taumeta_values, num_traj_values

    def helper(self, len_trajectory, num_trajectories):
        dataarray = np.asarray(self.simulated_data)
        return self.log_avg_error_naive(dataarray)

    """
    ######
    ### Naive Performance Calculation Methods (Taumeta)
    ######
    """

    def test_taumeta_eta_performance_only(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # ToDo Document The formulas used here need justification
                self.taumeta = taumeta
                self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                scale_window = Variable_Holder.mid_scale_window
                len_trajectory = Variable_Holder.len_trajectory_max
                self.window_size = scale_window * self.shift
                self.num_trajectories =  len(simulated_data)

                self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_total_time_naive = self.helper_performance_only(len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive

        return avg_times_naive, taumeta_values, eta_values

    def test_taumeta_scale_window_performance_only(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                len_trajectory = Variable_Holder.len_trajectory_max
                self.shift = eta * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories =  len(simulated_data)
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory_max, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations,len_trajectory, self.num_trajectories,
                                        eta,
                                        scale_window)

                log_total_time_naive = self.helper_performance_only(len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive

        return avg_times_naive, taumeta_values, scale_window_values

    def test_taumeta_num_traj_performance_only(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):
                self.taumeta = taumeta
                self.spectral_mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                scale_window = Variable_Holder.mid_scale_window
                self.shift = eta * self.taumeta
                self.window_size = scale_window * self.shift
                self.num_trajectories = num_traj
                len_trajectory = int(
                    Variable_Holder.num_trajectories_num_transitions_max / self.num_trajectories) + 1

                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_total_time_naive = self.helper_performance_only(len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive

        return avg_times_naive, taumeta_values, num_traj_values

    def helper_performance_only(self, len_trajectory, num_trajectories):
        dataarray = np.asarray(self.simulated_data)
        return self.log_avg_performance_naive(dataarray)

    """
    ######
    ### Performance and Error Utility Calculation Methods
    ######
    """

    def log_avg_performance_naive(self, dataarray):
        lag = int(Variable_Holder.max_taumeta / self.taumeta)
        etimenaive = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            t0 = process_time()
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # initialization - we found out that calling the count_matrix_coo2_mult function the first time results
                # in lower performance than for following calls - probably due to caching in the background. To avoid
                # this deviation, we call this function once - for starting the cache procedure.
                estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True,
                                             lag=lag)

            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                              initial=True)  # count matrix for whole window
            A0 = _tm(C0)
            etimenaive[k] = t0 - process_time()
        log_total_time_naive = Utility.log_value(etimenaive[-1])
        return log_total_time_naive

    def performance_naive(self, dataarray):
        lag = int(Variable_Holder.max_taumeta/self.taumeta)
        etimenaive = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            t0 = process_time()
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # initialization - we found out that calling the count_matrix_coo2_mult function the first time results
                # in lower performance than for following calls - probably due to caching in the background. To avoid
                # this deviation, we call this function once - for starting the cache procedure.
                estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True, lag=lag)

            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                              initial=True)  # count matrix for whole window
            A0 = _tm(C0)
            etimenaive[k] = t0 - process_time()
        return etimenaive

    def log_avg_error_naive(self, dataarray):
        lag = int(Variable_Holder.max_taumeta/self.taumeta)
        err = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # initialization - we found out that calling the count_matrix_coo2_mult function the first time results
                # in lower performance than for following calls - probably due to caching in the background. To avoid
                # this deviation, we call this function once - for starting the cache procedure.
                estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True, lag=lag)

            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                              initial=True)  # count matrix for whole window
            A0 = _tm(C0)
            err[k] = np.linalg.norm(A0 - self.spectral_mm1_0_0_scaled.trans)

        log_avg_err_naive = Utility.log_value(sum(err) / len(err))
        return log_avg_err_naive


    def error_naive(self, dataarray):
        lag = int(Variable_Holder.max_taumeta/self.taumeta)
        err = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # initialization - we found out that calling the count_matrix_coo2_mult function the first time results
                # in lower performance than for following calls - probably due to caching in the background. To avoid
                # this deviation, we call this function once - for starting the cache procedure.
                estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True, lag=lag)

            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                              initial=True)  # count matrix for whole window
            A0 = _tm(C0)
            err[k] = np.linalg.norm(A0 - self.spectral_mm1_0_0_scaled.trans)
        return err


    def print_param_values(self, evaluation_name, taumeta, shift, window_size, num_estimations, len_trajectory,
                           num_trajectories, eta, scale_window):
        print("Parameter Overview for " + evaluation_name + ":")
        print("taumeta:\t", taumeta)
        print("eta:\t", eta)
        print("scale_window\t:", scale_window)
        print("shift:\t", shift)
        print("window_size:\t", window_size)
        print("num_estimations:\t", num_estimations)
        print("len_trajectory:\t", len_trajectory)
        print("num_trajectories:\t", num_trajectories)
        print("num_trajectories*len_trajectory:\t", num_trajectories * len_trajectory)
        print("NAIVE window_size * num_estimations\t", window_size * (num_estimations + 1))
        print("BAYES window_size + num_estimations*shift\t", window_size + num_estimations * shift)
        print("\n")
