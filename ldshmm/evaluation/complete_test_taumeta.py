from time import process_time
from unittest import TestCase

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder


class Approach_Test(TestCase):
    def setUp(self):

        self.num_states = 4
        self.mmf1_0 = MMFamily1(self.num_states)
        self.mm1_0_0 = self.mmf1_0.sample()[0]

        self.min_eta=Variable_Holder.min_eta
        self.min_scale_window=Variable_Holder.min_scale_window
        self.min_num_traj=Variable_Holder.min_num_trajectories
        self.heatmap_size=Variable_Holder.heatmap_size
        self.min_taumeta=Variable_Holder.min_taumeta

        self.mid_eta = Variable_Holder.mid_eta
        self.mid_scale_window = Variable_Holder.mid_scale_window
        self.mid_num_traj = Variable_Holder.mid_num_trajectories
        self.mid_taumeta = Variable_Holder.mid_taumeta

        import math
        self.max_eta = self.min_eta* math.pow(2, self.heatmap_size-1)
        self.max_taumeta = self.min_taumeta* math.pow(2, self.heatmap_size-1)
        self.shift_max = self.max_eta * self.max_taumeta
        self.nwindow_max = self.mid_scale_window * self.shift_max
        self.num_estimations_max = 1 #smallest value within the heatmap

        self.num_trajectories_max = self.min_num_traj*math.pow(2, self.heatmap_size-1)

        self.len_trajectory = int(self.nwindow_max + self.num_estimations_max * self.shift_max +1)
        self.num_trajectories_len_trajectory_max = self.num_trajectories_max * self.len_trajectory
        """
        self.shift_mid = self.mid_eta*self.mid_taumeta
        self.nwindow_mid = self.mid_scale_window * self.shift_mid
        self.num_estimations_mid = Utility.calc_num_estimations_mid(self.nwindow_mid, self.heatmap_size, self.shift_mid)
        """

    def test_run_all_tests(self):
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

        # calculate performances and errors for the three parameters
        avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = self.test_taumeta_eta()

        avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = self.test_taumeta_scale_window()

        avg_times_naive3,  avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = self.test_taumeta_num_traj()

        # get minimum and maximum performance
        min_val = np.amin([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3])
        max_val = np.amax([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3])


        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_window_values, y_label="scale_window", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj", minimum=min_val, maximum=max_val)

        plots.save_plot_same_colorbar("Performance")

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)
        plots.add_to_plot_separate_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=eta_values, y_label="eta")
        plots.add_to_plot_separate_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_window_values, y_label="scale_window")
        plots.add_to_plot_separate_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj")
        plots.save_plot_separate_colorbars("Performance_separate_colorbars")
        ###########################################################self.num_estimations_mid = Utility.calc_num_estimations_mid(self.nwindow, self.heatmap_size, self.shift)

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=3)

        # get minimum and maximum error
        min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3])
        max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3])

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, x_labels=taumeta_values,
                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, x_labels=taumeta_values,
                            y_labels=scale_window_values, y_label="scale_window", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3, x_labels=taumeta_values,
                            y_labels=num_traj_values, y_label="num   _traj", minimum=min_val, maximum=max_val)


        plots.save_plot_same_colorbar("Error")
        ##########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=3)
        plots.add_to_plot_separate_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1,
                                            x_labels=taumeta_values, y_labels=eta_values, y_label="eta")
        plots.add_to_plot_separate_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2,
                                            x_labels=taumeta_values, y_labels=scale_window_values, y_label="scale_window")
        plots.add_to_plot_separate_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3,
                                            x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj")
        plots.save_plot_separate_colorbars("Error_separate_colorbars")
        ###########################################################

    def test_taumeta_eta(self):

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        eta_values = create_value_list(self.min_eta, self.heatmap_size)

        for one,taumeta in enumerate(taumeta_values):
            for two,eta in enumerate(eta_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                self.nwindow = self.mid_scale_window * self.shift
                self.num_trajectories = self.mid_num_traj

                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.nwindow, self.shift)
                self.r = (self.nwindow - self.shift) / self.nwindow

                self.print_param_values("ETA", self.taumeta, self.shift, self.nwindow, self.num_estimations, self.len_trajectory, self.num_trajectories, eta, self.mid_scale_window)

                slope_time_naive, avg_err_naive,  slope_time_bayes, avg_err_bayes  = self.test_eta_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes


        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, eta_values

    def test_eta_helper(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(
                dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            return self.test_eta_helper()


    def test_taumeta_scale_window(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        scale_window_values = create_value_list(self.min_scale_window,self.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (self.mid_eta) * self.taumeta
                self.nwindow = scale_window * self.shift
                self.num_trajectories = self.mid_num_traj
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.nwindow, self.shift)
                self.r = (self.nwindow - self.shift) / self.nwindow

                self.print_param_values("scale_window", self.taumeta, self.shift, self.nwindow, self.num_estimations, self.len_trajectory, self.num_trajectories, self.mid_eta, scale_window)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes  = self.test_scale_window_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, scale_window_values

    def test_scale_window_helper(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(
            dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            return self.test_scale_window_helper()

    def test_taumeta_num_traj(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        num_traj_values = create_value_list(self.min_num_traj,self.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (self.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.nwindow = (self.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(self.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.nwindow, self.shift)
                self.r = (self.nwindow - self.shift) / self.nwindow

                self.print_param_values("NUM_TRAJ",self.taumeta, self.shift, self.nwindow, self.num_estimations, self.len_trajectory, self.num_trajectories, self.mid_eta, self.mid_scale_window)

                slope_time_naive, avg_err_naive,  slope_time_bayes, avg_err_bayes  = self.test_num_traj_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)
        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, num_traj_values

    def test_num_traj_helper(self):
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            return self.test_num_traj_helper()

    def performance_and_error_calculation(self, dataarray, err, errbayes, etimebayes, etimenaive):
        for k in range(0, self.num_estimations + 1):
            assert (self.nwindow + k * self.shift) < np.shape(dataarray)[1]
            data0 = dataarray[:, k * self.shift: (self.nwindow + k * self.shift)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            t0 = process_time()
            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)  # count matrix for whole window
            C0 += 1e-8
            t1 = process_time()
            A0 = _tm(C0)

            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            err[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.nwindow + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)
                etimebayes[1] = process_time() - t0
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.nwindow + (k - 1) * self.shift - 1: (self.nwindow + k * self.shift)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                t0 = process_time()
                C_new = estimate_via_sliding_windows(data=dataslice1new, num_states=self.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)
        print("Sum Err Naive:", Utility.log_value(sum(err)/len(err)))
        print("Sum Err Bayes:", Utility.log_value(sum(errbayes/len(err))))

        slope_time_naive = Utility.log_value(etimenaive[-1])
        avg_err_naive = Utility.log_value(sum(err) / len(err))
        slope_time_bayes = Utility.log_value(etimebayes[-1])
        avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))

        return slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes

    def print_param_values(self, evaluation_name, taumeta, shift, nwindow, num_estimations, len_trajectory, num_trajectories, eta, scale_window):
        print("Parameter Overview for " + evaluation_name+ ":")
        print("taumeta:\t", taumeta)
        print("eta:\t", eta)
        print("scale_window\t:", scale_window)
        print("shift:\t", shift)
        print("nwindow:\t", nwindow)
        print("num_estimations:\t", num_estimations)
        print("len_trajectory:\t", len_trajectory)
        print("num_trajectories:\t", num_trajectories)
        print("num_trajectories*len_trajectory:\t", num_trajectories*len_trajectory)
        print("\n")
