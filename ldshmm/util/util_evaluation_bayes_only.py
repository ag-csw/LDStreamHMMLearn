from time import process_time

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder

class Evaluation_Holder():
    """
    Class holding all evaluation functions used for a ConvexCombinationQuasiMM.
    """

    #TODO adapt Evaluate_Delta to avoid duplicate codes

    def __init__(self, qmm1_0_0, delta, simulate=True):
        """
        :param qmm1_0_0: ConvexCombinationQuasiMM (for instance obtained by sampling the QMMFamily1)
        :param delta: int or float
        :param simulate: bool (default=True) - decide whether data should be simulated in the constructor
        """

        self.qmm1_0_0 = qmm1_0_0
        if simulate:
            simulate_and_store_data(qmm1_0_0, "qmm")
            self.simulated_data = read_simulated_data("qmm")
        self.delta = delta
        self.timescaledisp = Variable_Holder.min_timescaledisp
        self.statconc = Variable_Holder.mid_statconc


    def test_taumeta_timescaledisp(self):
        #FIXME: change test so that timescaledisp is on the horizontal axis and eta is on the vertical axis
        # Also, let's move this to a separate test, where we have timescaledisp on the horizontal axis for
        # all heatmaps, and the vertical axes are the same as the tests above (eta, ..., numtraj)
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        timescaledisp_values = create_value_list(Variable_Holder.min_timescaledisp, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, timescaledisp in enumerate(timescaledisp_values):
                # ToDo Document
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=timescaledisp, statconc=Variable_Holder.mid_statconc)
                self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta=self.delta)
                self.qmm1_0_0 = self.qmmf1_0.sample()[0]

                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = Variable_Holder.mid_eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("TIMESCALEDISP", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta,
                                        Variable_Holder.mid_scale_window, timescaledisp, Variable_Holder.mid_statconc)

                slope_time_naive, log_avg_err_naive, slope_time_bayes, log_avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, timescaledisp_values

    def test_taumeta_statconc(self):
        #FIXME: change test so that statconc is on the horizontal axis and eta is on the vertical axis
        # Also, let's move this to a separate test, where we have statconc on the horizontal axis for
        # all heatmaps, and the vertical axes are the same as the tests above (eta, ..., numtraj)
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        statconc_values = Variable_Holder.statconc_values  # create_value_list_floats(Variable_Holder.min_statconc, self.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, statconc in enumerate(statconc_values):
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=Variable_Holder.mid_timescaledisp, statconc=statconc)
                self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta=self.delta)
                self.qmm1_0_0 = self.qmmf1_0.sample()[0]

                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = Variable_Holder.mid_eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("STATCONC", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta,
                                        Variable_Holder.mid_scale_window, Variable_Holder.mid_timescaledisp, statconc)

                slope_time_naive, log_avg_err_naive, slope_time_bayes, log_avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, statconc_values


    """
    ######
    ### Bayes Error Calculation Methods (Taumeta)
    #####
    """

    def test_taumeta_eta(self, qmm1_0_0=None, simulated_data=None):
        if qmm1_0_0:
            self.qmm1_0_0 = qmm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                # ToDo Document The formulas used here need justification
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                scale_window = Variable_Holder.mid_scale_window
                len_trajectory = Variable_Holder.len_trajectory_max
                self.window_size = scale_window * self.shift
                self.num_trajectories = len(simulated_data)

                self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_avg_err_bayes = self.helper(len_trajectory, self.num_trajectories)

                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_errs_bayes, taumeta_values, eta_values

    def test_taumeta_scale_window(self, qmm1_0_0=None, simulated_data=None):
        if qmm1_0_0:
            self.qmm1_0_0 = qmm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                self.shift = eta * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = len(simulated_data)
                len_trajectory = Variable_Holder.len_trajectory_max
                self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_avg_err_bayes = self.helper(len_trajectory, self.num_trajectories)

                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_errs_bayes, taumeta_values, scale_window_values

    def test_taumeta_num_traj(self, qmm1_0_0=None, simulated_data=None):
        if qmm1_0_0:
            self.qmm1_0_0 = qmm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                self.shift = eta * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                scale_window = Variable_Holder.min_scale_window
                self.window_size = scale_window * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(
                    Variable_Holder.num_trajectories_num_transitions_max / self.num_trajectories) + 1
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, eta, scale_window)

                log_avg_err_bayes = self.helper(self.len_trajectory, self.num_trajectories)

                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_errs_bayes, taumeta_values, num_traj_values

    def helper(self, len_trajectory, num_trajectories):
        dataarray = np.asarray(self.simulated_data[self.taumeta])
        return self.error_bayes(dataarray)

    """
    ######
    ### Bayes Performance Calculation Methods (Taumeta)
    ######
    """

    def test_taumeta_eta_performance_only(self, qmm1_0_0=None, simulated_data=None):
        if qmm1_0_0:
            self.qmm1_0_0 = qmm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                # ToDo Document The formulas used here need justification
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                scale_window = Variable_Holder.mid_scale_window
                self.window_size = scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories
                len_trajectory = Variable_Holder.len_trajectory_max

                self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_total_time_bayes = self.helper_performance_only(len_trajectory,
                                                                    self.num_trajectories)

                avg_times_bayes[two][one] = log_total_time_bayes

        return avg_times_bayes, taumeta_values, eta_values

    def test_taumeta_scale_window_performance_only(self, qmm1_0_0=None, simulated_data=None):
        if qmm1_0_0:
            self.qmm1_0_0 = qmm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                self.shift = eta * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = len(simulated_data)
                len_trajectory = Variable_Holder.len_trajectory_max
                self.num_estimations = Utility.calc_num_estimations(len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, len_trajectory, self.num_trajectories,
                                        eta,
                                        scale_window)

                log_total_time_bayes = self.helper_performance_only(len_trajectory, self.num_trajectories)

                avg_times_bayes[two][one] = log_total_time_bayes

        return avg_times_bayes, taumeta_values, scale_window_values

    def test_taumeta_num_traj_performance_only(self, qmm1_0_0=None, simulated_data=None):
        if qmm1_0_0:
            self.qmm1_0_0=qmm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                eta = Variable_Holder.mid_eta
                self.shift = eta * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                scale_window = Variable_Holder.mid_scale_window
                self.window_size = scale_window * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(
                    Variable_Holder.num_trajectories_num_transitions_max / self.num_trajectories) + 1
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, eta,
                                        scale_window)

                log_total_time_bayes = self.helper_performance_only(self.len_trajectory, self.num_trajectories)

                avg_times_bayes[two][one] = log_total_time_bayes

        return avg_times_bayes, taumeta_values, num_traj_values

    def helper_performance_only(self, len_trajectory, num_trajectories):
        dataarray = np.asarray(self.simulated_data[self.taumeta])
        return self.performance_bayes(dataarray)

    """
    ######
    ### Bayes Performance Calculation Methods (Taumeta)
    ######
    """

    def performance_and_error_calculation_old(self, dataarray):
        """
        Method to perform (num_estimations+1) timing and error calculations.
        We return the latest log timing calculation and the average log error calculation for the naive and bayes approach.

        :param dataarray: ndarray - trajectory to perform timings and calculate errors from
        :return: tuple of (slope_time_naive, log_avg_err_naive, slope_time_bayes, log_avg_err_bayes)
        """

        try:
            etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
            etimenaive[0] = 0
            err = np.zeros(self.num_estimations + 1, dtype=float)
            etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
            errbayes = np.zeros(self.num_estimations + 1, dtype=float)

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
                    estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states)

                t0 = process_time()
                C0 = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True)  # count matrix for whole window
                t1 = process_time()
                A0 = _tm(C0)

                etimenaive[k + 1] = t1 - t0 + etimenaive[k]
                err[k] = np.linalg.norm(A0 - self.qmm1_0_0_scaled.eval(k).trans)
                if k == 0:
                    ##### Bayes approach: Calculate C0 separately
                    data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                    dataslice0 = []
                    for i in range(0, self.num_trajectories):
                        dataslice0.append(data0[i, :])

                    t0 = process_time()
                    C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True)
                    etimebayes[1] = process_time() - t0
                    errbayes[0] = np.linalg.norm(_tm(C_old) - self.qmm1_0_0_scaled.eval(k).trans)

                if k >= 1:
                    ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                    data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                    dataslice1new = []
                    for i in range(0, self.num_trajectories):
                        dataslice1new.append(data1new[i, :])
                    t0 = process_time()
                    C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                         num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                    weight0 = self.r
                    weight1 = 1.0

                    C1bayes = weight0 * C_old + weight1 * C_new
                    C_old = C1bayes

                    t1 = process_time()
                    etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                    A1bayes = _tm(C1bayes)
                    errbayes[k] = np.linalg.norm(A1bayes - self.qmm1_0_0_scaled.eval(k).trans)

            slope_time_naive = Utility.log_value(etimenaive[-1])
            log_avg_err_naive = Utility.log_value(sum(err) / len(err))
            slope_time_bayes = Utility.log_value(etimebayes[-1])
            log_avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))

            return slope_time_naive, log_avg_err_naive, slope_time_bayes, log_avg_err_bayes
        except:
            print("Exception occured")

    def performance_and_error_calculation(self, dataarray):
        # ToDo Document

        log_total_time_naive, log_avg_err_naive = self.performance_and_error_calculation_naive(dataarray)
        log_total_time_bayes, log_avg_err_bayes = self.performance_and_error_calculation_bayes(dataarray)

        return log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes

    def performance_and_error_calculation_naive(self, dataarray):
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            t0 = process_time()
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])

            C0 = estimate_via_sliding_windows(data=dataslice0,
                                              num_states=Variable_Holder.num_states,
                                              initial=True)  # count matrix for whole window
            A0 = _tm(C0)
            t1 = process_time()

            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            err[k] = np.linalg.norm(A0 - self.qmm1_0_0_scaled.eval(k).trans)

        log_total_time_naive = Utility.log_value(etimenaive[-1])
        log_avg_err_naive = Utility.log_value(sum(err) / len(err))

        return log_total_time_naive, log_avg_err_naive

    def performance_and_error_calculation_bayes(self, dataarray):
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                t0 = process_time()
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                                     initial=True)
                A0 = _tm(C_old)
                etimebayes[1] = process_time() - t0
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.qmm1_0_0_scaled.eval(k).trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                t0 = process_time()
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                errbayes[k] = np.linalg.norm(A1bayes - self.qmm1_0_0_scaled.eval(k).trans)

        log_total_time_bayes = Utility.log_value(etimebayes[-1])
        log_avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))
        return log_total_time_bayes, log_avg_err_bayes

    def performance_bayes(self, dataarray):
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                t0 = process_time()
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                                     initial=True)
                A0 = _tm(C_old)
                etimebayes[1] = process_time() - t0

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                t0 = process_time()
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]

        log_total_time_bayes = Utility.log_value(etimebayes[-1])
        return log_total_time_bayes

    def error_bayes(self, dataarray):
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states,
                                                     initial=True)
                A0 = _tm(C_old)
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.qmm1_0_0_scaled.eval(k).trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.qmm1_0_0_scaled.eval(k).trans)

        log_avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))
        return log_avg_err_bayes

    def print_param_values(self, evaluation_name, taumeta, shift, window_size, num_estimations, len_trajectory,
                           num_trajectories, eta, scale_window, timescaledisp=None, statconc=None):
        """
        Utility method to print parameter values within the evaluation procedure.

        :param evaluation_name: str - name of the evaluation or parameter to evaluate
        :param taumeta: int
        :param shift: int
        :param window_size: int
        :param num_estimations: int
        :param len_trajectory: int
        :param num_trajectories: int
        :param eta: int
        :param scale_window: int
        :param timescaledisp: int
        :param statconc: float
        """

        print("Parameter Overview for " + evaluation_name + ":")
        print("taumeta:\t", taumeta)
        print("eta:\t", eta)
        print("scale_window\t:", scale_window)
        print("shift:\t", shift)
        print("window_size:\t", window_size)
        print("num_estimations:\t", num_estimations)
        print("len_trajectory:\t", len_trajectory)
        print("num_trajectories:\t", num_trajectories)
        print("timescaledisp:\t", timescaledisp)
        print("statconc:\t", statconc)
        print("num_trajectories*len_trajectory:\t", num_trajectories * len_trajectory)
        print("NAIVE window_size * num_estimations\t", window_size * (num_estimations + 1))
        print("BAYES window_size + num_estimations*shift\t", window_size + num_estimations * shift)
        print("\n")


