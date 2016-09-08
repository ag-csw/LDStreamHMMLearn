from unittest import TestCase
import numpy as np
import pyemma.msm as MSM
import pyemma.msm.estimators as _MME
from msmtools.estimation import transition_matrix as _tm
from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from time import process_time
import matplotlib.pyplot as plt
from ldshmm.test.plottings import plot_result_heatmap

from ldshmm.util.mm_family import MMFamily1

class Approach_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]
        self.numsteps = 100
        self.ntraj = 20

    def estimate_via_sliding_windows(self, data):
        C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates=self.nstates)
        return C

    def test_taumeta_scale_win(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.numsteps + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.numsteps + 1, dtype=float)

        etimebayes = np.zeros(self.numsteps + 2, dtype=float)
        errbayes = np.zeros(self.numsteps + 1, dtype=float)

        # initialize average timing and error arrays for naive and bayes
        avg_times_naive = np.zeros((3, 3))
        avg_errs_naive = np.zeros((3, 3))
        avg_times_bayes = np.zeros((3, 3))
        avg_errs_bayes = np.zeros((3, 3))

        # specify values for taumeta and eta to iterate over
        taumeta_values = [2, 4, 8]  # 2,4,8
        scale_win_values = [25, 50, 100]  # 50,100,200

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_win in enumerate(scale_win_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.nstep = 100 * self.taumeta
                self.nwindow = scale_win * self.nstep
                self.numsteps = 100
                self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
                self.ntraj = 20
                self.r = (self.nwindow - self.nstep) / self.nwindow

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

                avg_time = sum(etimenaive) / len(etimenaive)
                avg_err = sum(err) / len(err)

                avg_times_naive[one][two] = avg_time
                avg_errs_naive[one][two] = avg_err

                avg_time_bayes = sum(etimebayes) / len(etimebayes)
                avg_err_bayes = sum(errbayes) / len(errbayes)

                avg_times_bayes[one][two] = avg_time_bayes
                avg_errs_bayes[one][two] = avg_err_bayes

        print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)

        # plot the average performances and errors in a heatmap
        plot_result_heatmap(avg_times_naive, avg_times_bayes, taumeta_values, scale_win_values, "scalewin", "Performance", "Heatmap Performance Taumeta ScaleWin")
        plot_result_heatmap(avg_errs_naive, avg_errs_bayes, taumeta_values, scale_win_values, "scalewin", "Error", "Heatmap Error Taumeta ScaleWin")
