from time import process_time
from unittest import TestCase

import numpy as np
from msmtools.estimation import transition_matrix as _tm
from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_math import Utility


class Approach_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]

        self.min_eta = 64
        self.min_scale_win = 16
        self.min_num_traj = 16

        self.heatmap_size = 3

    def estimate_via_sliding_windows(self, data):
        C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates=self.nstates)
        return C

    def test_taumeta_numsteps(self):
        # initialize timing and error arrays for naive and bayes

        # initialize average timing and error arrays for naive and bayes
        avg_times_naive = np.zeros((self.heatmap_size, self.heatmap_size))
        avg_errs_naive = np.zeros((self.heatmap_size, self.heatmap_size))
        avg_times_bayes = np.zeros((self.heatmap_size, self.heatmap_size))
        avg_errs_bayes = np.zeros((self.heatmap_size, self.heatmap_size))

        # specify values for taumeta and eta to iterate over
        taumeta_values = [2, 4, 8]
        numsteps_values = [64,128,256]

        for one, taumeta in enumerate(taumeta_values):
            for two, numstep in enumerate(numsteps_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.nstep = (self.min_eta*2) * self.taumeta
                self.nwindow = (self.min_scale_win*2) * self.nstep
                self.numsteps = numstep
                self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
                self.ntraj = (self.min_num_traj*2)
                self.r = (self.nwindow - self.nstep) / self.nwindow

                etimenaive = np.zeros(self.numsteps + 2, dtype=float)
                etimenaive[0] = 0
                err = np.zeros(self.numsteps + 1, dtype=float)

                etimebayes = np.zeros(self.numsteps + 2, dtype=float)
                errbayes = np.zeros(self.numsteps + 1, dtype=float)

                self.data1_0_0 = []
                for i in range(0, self.ntraj):
                    self.data1_0_0.append(self.mm1_0_0_scaled.simulate(self.lentraj))
                dataarray = np.asarray(self.data1_0_0)

                # do the timing and error calculation (numsteps+1)- times and calculate the average from these
                for k in range(0, self.numsteps + 1):

                    ##### naive sliding window approach
                    data0 = dataarray[:, k * self.nstep: (self.nwindow + k * self.nstep)]
                    dataslice0 = []
                    for i in range(0, self.ntraj):
                        dataslice0.append(data0[i, :])
                    t0 = process_time()
                    C0 = self.estimate_via_sliding_windows(dataslice0)  # count matrix for whole window
                    t1 = process_time()
                    A0 = _tm(C0)
                    etimenaive[k + 1] = t1 - t0 + etimenaive[k]
                    err[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)
                    if k == 0:
                        ##### Bayes approach: Calculate C0 separately
                        data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
                        dataslice0 = []
                        for i in range(0, self.ntraj):
                            dataslice0.append(data0[i, :])

                        t0 = process_time()
                        C_old = self.estimate_via_sliding_windows(dataslice0)
                        etimebayes[1] = process_time() - t0
                        errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

                    if k >= 1:
                        ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                        data1new = dataarray[:, self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
                        dataslice1new = []
                        for i in range(0, self.ntraj):
                            dataslice1new.append(data1new[i, :])
                        t0 = process_time()
                        C_new = self.estimate_via_sliding_windows(
                            dataslice1new)  # count matrix for just new transitions

                        weight0 = self.r
                        weight1 = 1.0

                        C1bayes = weight0 * C_old + weight1 * C_new
                        C_old = C1bayes

                        t1 = process_time()
                        etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                        A1bayes = _tm(C1bayes)
                        errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)

                slope_time_naive = Utility.log_value(Utility.calc_slope(etimenaive))
                avg_err_naive = Utility.log_value(sum(err) / len(err))
                slope_time_bayes = Utility.log_value(Utility.calc_slope(etimebayes))
                avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))

                avg_times_naive[one][two] = slope_time_naive
                avg_errs_naive[one][two] = avg_err_naive

                avg_times_bayes[one][two] = slope_time_bayes
                avg_errs_bayes[one][two] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        min_perf =  np.amin([avg_times_naive, avg_times_bayes])
        max_perf = np.amax([avg_times_naive, avg_times_bayes])
        min_err = np.amin([avg_errs_naive, avg_errs_bayes])
        max_err = np.amax([avg_errs_naive, avg_errs_bayes])

        # plot the average performances and errors in a heatmap
        plot = ComplexPlot()
        plot.new_plot("Taumeta Numsteps Stationary MM Performance", rows=1)
        plot.add_to_plot_same_colorbar(data_naive=avg_times_naive, data_bayes=avg_times_bayes, y_labels=numsteps_values, x_labels=taumeta_values, y_label="Numsteps", minimum=min_perf, maximum=max_perf)
        plot.save_plot_same_colorbar(heading="Taumeta Numsteps Performance")

        plot = ComplexPlot()
        plot.new_plot("Taumeta Numsteps Stationary MM Errors", rows=1)
        plot.add_to_plot_same_colorbar(data_naive=avg_errs_naive, data_bayes=avg_errs_bayes, y_labels=numsteps_values,
                                       x_labels=taumeta_values, y_label="Numsteps", minimum=min_err, maximum=max_err)
        plot.save_plot_same_colorbar(heading="Taumeta Numsteps Error")