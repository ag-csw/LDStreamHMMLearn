import math
from time import process_time
from unittest import TestCase
from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.util_functionality import *


class Approach_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.timescaledisp = 2.0
        self.statconc = 0.05
        self.mmf1_0 = MMFamily1(self.nstates, self.timescaledisp, self.statconc)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0)
        self.qmm1_0_0 = self.qmmf1_0.sample()[0]
        self.heatmap_size = Variable_Holder.heatmap_size
        self.min_nu = Variable_Holder.min_nu
        self.min_taumeta = Variable_Holder.min_taumeta
        self.min_scale_win = Variable_Holder.min_scale_win
        self.min_num_traj = Variable_Holder.min_num_traj
        self.mid_nu = Variable_Holder.mid_nu
        self.mid_scale_win = Variable_Holder.mid_scale_win
        self.mid_num_traj = Variable_Holder.mid_num_traj


    def test_run_all_tests(self):
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

        # calculate performances and errors for the three parameters
        avg_times_naive1, avg_times_bayes1, avg_errs_naive1, avg_errs_bayes1, taumeta_values, nu_values = self.test_taumeta_nu()

        avg_times_naive2, avg_times_bayes2, avg_errs_naive2, avg_errs_bayes2, taumeta_values, scale_win_values = self.test_taumeta_scale_win()

        avg_times_naive3, avg_times_bayes3, avg_errs_naive3, avg_errs_bayes3, taumeta_values, num_traj_values = self.test_taumeta_numtraj()

        # get minimum and maximum performance
        min_val = np.amin([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3])
        print([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3], "Minimum:", min_val)

        max_val = np.amax([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3])
        print([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
               avg_times_bayes3], "Maximum:", max_val)

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=nu_values, y_label="nu", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_win_values, y_label="scale_win", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj", minimum=min_val, maximum=max_val)

        plots.save_plot_same_colorbar("Performance-nonstat")

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)
        plots.add_to_plot_separate_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=nu_values, y_label="nu")
        plots.add_to_plot_separate_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_win_values, y_label="scale_win")
        plots.add_to_plot_separate_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj")
        plots.save_plot_separate_colorbars("Performance_separate_colorbars-nonstat")
        ###########################################################

        ###########################################################

        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=3)

        # get minimum and maximum error
        min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3])
        print([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3], "Minimum:", min_val)


        max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3])
        print([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
               avg_errs_bayes3], "Maximum:", max_val)

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, x_labels=taumeta_values,
                            y_labels=nu_values, y_label="nu", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, x_labels=taumeta_values,
                            y_labels=scale_win_values, y_label="scale_win", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3, x_labels=taumeta_values,
                            y_labels=num_traj_values, y_label="num_traj", minimum=min_val, maximum=max_val)


        plots.save_plot_same_colorbar("Error-nonstat")
        ##########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=3)
        plots.add_to_plot_separate_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1,
                                            x_labels=taumeta_values, y_labels=nu_values, y_label="nu")
        plots.add_to_plot_separate_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2,
                                            x_labels=taumeta_values, y_labels=scale_win_values, y_label="scale_win")
        plots.add_to_plot_separate_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3,
                                            x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj")
        plots.save_plot_separate_colorbars("Error_separate_colorbars-nonstat")
        ###########################################################

    def test_taumeta_scale_win(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        scale_win_values = create_value_list(self.min_scale_win, self.heatmap_size)  # 5,10,20

        for one,taumeta in enumerate(taumeta_values):
            for two,scale_win in enumerate(scale_win_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.tauquasi = self.timescaledisp * 3
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta, self.tauquasi)

                self.nu = self.mid_nu
                self.nstep = math.ceil(self.nu * self.timescaledisp * self.taumeta * self.tauquasi)
                self.nwindow = scale_win * self.nstep
                self.numsteps = int(Variable_Holder.numsteps_global_nonstat / Variable_Holder.product_mid_values_nonstat)
                self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
                self.ntraj = self.mid_num_traj
                self.r = (self.nwindow - self.nstep) / self.nwindow


                etimenaive = np.zeros(self.numsteps + 2, dtype=float)
                etimenaive[0] = 0
                err = np.zeros(self.numsteps + 1, dtype=float)

                etimebayes = np.zeros(self.numsteps + 2, dtype=float)
                errbayes = np.zeros(self.numsteps + 1, dtype=float)

                self.data1_0_0 = []
                for i in range(0, self.ntraj):
                    self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(self.lentraj))
                dataarray = np.asarray(self.data1_0_0)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.performance_and_error_calculation(
                    dataarray, err, errbayes, etimebayes, etimenaive)
                avg_times_naive[one][two] = slope_time_naive
                avg_errs_naive[one][two] = avg_err_naive

                avg_times_bayes[one][two] = slope_time_bayes
                avg_errs_bayes[one][two] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        return avg_times_naive, avg_times_bayes, avg_errs_naive, avg_errs_bayes, taumeta_values, scale_win_values

    def test_taumeta_nu(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        nu_values = create_value_list(self.min_nu, self.heatmap_size) #5,10,25

        for one, taumeta in enumerate(taumeta_values):
            for two, nu in enumerate(nu_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.tauquasi = self.timescaledisp * 3
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta, self.tauquasi)
                self.nu = nu
                self.nstep = math.ceil(self.nu * self.timescaledisp * self.taumeta * self.tauquasi)
                self.nwindow = self.mid_scale_win * self.nstep
                self.numsteps = int(Variable_Holder.numsteps_global_nonstat / Variable_Holder.product_mid_values_nonstat)
                self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
                self.ntraj = self.mid_num_traj


                self.r = (self.nwindow - self.nstep) / self.nwindow

                etimenaive = np.zeros(self.numsteps + 2, dtype=float)
                etimenaive[0] = 0
                err = np.zeros(self.numsteps + 1, dtype=float)

                etimebayes = np.zeros(self.numsteps + 2, dtype=float)
                errbayes = np.zeros(self.numsteps + 1, dtype=float)

                self.data1_0_0 = []
                for i in range(0, self.ntraj):
                    self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(self.lentraj))
                dataarray = np.asarray(self.data1_0_0)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.performance_and_error_calculation(dataarray, err, errbayes, etimebayes, etimenaive)
                avg_times_naive[one][two] = slope_time_naive
                avg_errs_naive[one][two] = avg_err_naive

                avg_times_bayes[one][two] = slope_time_bayes
                avg_errs_bayes[one][two] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        return avg_times_naive, avg_times_bayes, avg_errs_naive, avg_errs_bayes, taumeta_values, nu_values

    def test_taumeta_numtraj(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        num_traj_values = create_value_list(self.min_num_traj, self.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.tauquasi = self.timescaledisp * 3
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta, self.tauquasi)

                self.nu = self.mid_nu
                self.nstep = math.ceil(self.nu * self.timescaledisp * self.taumeta * self.tauquasi)
                self.nwindow = self.min_scale_win * self.nstep
                self.numsteps = int(Variable_Holder.numsteps_global_nonstat / Variable_Holder.product_mid_values_nonstat)
                self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
                self.ntraj = num_traj

                self.r = (self.nwindow - self.nstep) / self.nwindow

                etimenaive = np.zeros(self.numsteps + 2, dtype=float)
                etimenaive[0] = 0
                err = np.zeros(self.numsteps + 1, dtype=float)

                etimebayes = np.zeros(self.numsteps + 2, dtype=float)
                errbayes = np.zeros(self.numsteps + 1, dtype=float)

                self.data1_0_0 = []
                for i in range(0, self.ntraj):
                    self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(self.lentraj))
                dataarray = np.asarray(self.data1_0_0)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes = self.performance_and_error_calculation(
                    dataarray, err, errbayes, etimebayes, etimenaive)
                avg_times_naive[one][two] = slope_time_naive
                avg_errs_naive[one][two] = avg_err_naive

                avg_times_bayes[one][two] = slope_time_bayes
                avg_errs_bayes[one][two] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        return avg_times_naive, avg_times_bayes, avg_errs_naive, avg_errs_bayes, taumeta_values, num_traj_values

    def performance_and_error_calculation(self, dataarray, err, errbayes, etimebayes, etimenaive):
        # do the timing and error calculation (numsteps+1)- times and calculate the average from these
        for k in range(0, self.numsteps + 1):

            ##### naive sliding window approach
            data0 = dataarray[:, k * self.nstep: (self.nwindow + k * self.nstep)]
            dataslice0 = []
            for i in range(0, self.ntraj):
                dataslice0.append(data0[i, :])
            t0 = process_time()
            C0 = estimate_via_sliding_windows(dataslice0, self.nstates)  # count matrix for whole window
            t1 = process_time()
            A0 = _tm(C0)
            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            err[k] = np.linalg.norm(A0 - self.qmm1_0_0_scaled.eval(self.nwindow + (k - 0.5) * self.nstep).trans)

            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
                dataslice0 = []
                for i in range(0, self.ntraj):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(dataslice0, self.nstates)
                etimebayes[1] = process_time() - t0
                errbayes[0] = np.linalg.norm(
                    _tm(C_old) - self.qmm1_0_0_scaled.eval(self.nwindow + (k - 0.5) * self.nstep).trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:,
                           self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
                dataslice1new = []
                for i in range(0, self.ntraj):
                    dataslice1new.append(data1new[i, :])
                t0 = process_time()
                C_new = estimate_via_sliding_windows(dataslice1new,
                                                     self.nstates)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(
                A1bayes - self.qmm1_0_0_scaled.eval(self.nwindow + (k - 0.5) * self.nstep).trans)
        slope_time_naive = Utility.log_value(Utility.calc_slope(etimenaive))
        avg_err_naive = Utility.log_value(sum(err) / len(err))
        slope_time_bayes = Utility.log_value(Utility.calc_slope(etimebayes))
        avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))
        return slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes
