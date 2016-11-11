from time import process_time

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder

class Evaluation_Holder():
    # ToDo Document

    def __init__(self, qmm1_0_0, delta, simulate=True):
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

        # specify values for taumeta and eta to iterate over
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

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, timescaledisp_values

    def test_taumeta_statconc(self):
        #FIXME: change test so that statconc is on the horizontal axis and eta is on the vertical axis
        # Also, let's move this to a separate test, where we have statconc on the horizontal axis for
        # all heatmaps, and the vertical axes are the same as the tests above (eta, ..., numtraj)
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for the concentration parameter and eta to iterate over
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

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, statconc_values

    def test_taumeta_eta(self, simulated_data=None):
        if simulated_data:
            self.simulated_data = simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for two, eta in enumerate(eta_values):
            for one, taumeta in enumerate(taumeta_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, eta, Variable_Holder.mid_scale_window,
                                        self.timescaledisp, self.statconc)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, eta_values

    def test_taumeta_scale_window(self, simulated_data=None):
        if simulated_data:
            self.simulated_data = simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for two, scale_window in enumerate(scale_window_values):
            for one, taumeta in enumerate(taumeta_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                self.window_size = scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, scale_window,
                                        self.timescaledisp, self.statconc)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, scale_window_values

    def helper(self, len_trajectory, num_trajectories):
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        dataarray = np.asarray(self.simulated_data[self.taumeta])
        dataarray = dataarray[:num_trajectories]
        dataarray = np.asarray([ndarr[:len_trajectory] for ndarr in dataarray])
        print(dataarray)
        try:
            return self.performance_and_error_calculation(
                dataarray, err, errbayes, etimebayes, etimenaive)
        except Exception as e:
            print("Exception thrown:",e )
            return self.helper(len_trajectory, num_trajectories)

    def test_taumeta_num_traj(self, simulated_data=None):
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)
        for two, num_traj in enumerate(num_traj_values):
            for one, taumeta in enumerate(taumeta_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.window_size = (Variable_Holder.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(Variable_Holder.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, Variable_Holder.mid_scale_window,
                                        self.timescaledisp, self.statconc)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.helper(self.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""
        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, num_traj_values

    def performance_and_error_calculation(self, dataarray, err, errbayes, etimebayes, etimenaive):
        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # init
                estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states)

            t0 = process_time()
            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states)  # count matrix for whole window
            C0 += 1e-8
            #print(C0)
            t1 = process_time()
            A0 = _tm(C0)
            #print("A0", A0)

            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            err[k] = np.linalg.norm(A0 - self.qmm1_0_0_scaled.eval(k).trans)
            #print("ErrNaive", err[k])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states)
                C0 += 1e-8
                etimebayes[1] = process_time() - t0
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.qmm1_0_0_scaled.eval(k).trans)
                #print("ErrBayes", errbayes[0])

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
                #print("C_old",C_old)

                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                A1bayes = _tm(C1bayes)
                #print("A1bayes",A1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.qmm1_0_0_scaled.eval(k).trans)
                #print("ErrBayes", errbayes[k])

        slope_time_naive = Utility.log_value(etimenaive[-1])
        avg_err_naive = Utility.log_value(sum(err) / len(err))
        slope_time_bayes = Utility.log_value(etimebayes[-1])
        avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))

        return slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes



    def print_param_values(self, evaluation_name, taumeta, shift, window_size, num_estimations, len_trajectory,
                           num_trajectories, eta, scale_window, timescaledisp, statconc):
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
