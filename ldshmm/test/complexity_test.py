from unittest import TestCase
import numpy as np
from msmtools.estimation import transition_matrix as _tm
from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from time import process_time
from ldshmm.util.mm_family import MMFamily1

class Complexity_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]
        self.numsteps = 100
        self.ntraj = 20

    def estimate_via_sliding_windows(self, data):
        C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates = self.nstates)
        return C

    def test_complexity(self):
        etimenaive = np.zeros(self.numsteps + 2, dtype=float)
        etimenaive[0] = 0
        etimebayes = np.zeros(self.numsteps + 2, dtype=float)

        # specify values for taumeta to iterate over - taumeta influences nsteps and therefore the datacliece size
        taumeta_values = [2,4,8]

        for one,taumeta in enumerate(taumeta_values):
            # Setting taumeta and eta values and recalculate dependent variables for scaling
            self.taumeta = taumeta
            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
            self.nstep = 100 * self.taumeta
            self.nwindow = 10 * self.nstep
            self.numsteps = 100
            self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
            self.ntraj = 20
            self.r = (self.nwindow - self.nstep) / self.nwindow

            self.data1_0_0 = []
            for i in range(0, self.ntraj):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(self.lentraj))
            dataarray = np.asarray(self.data1_0_0)

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


                if k==0:
                    ##### Bayes approach: Calculate C0 separately
                    data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
                    dataslice0 = []
                    for i in range(0, self.ntraj):
                        dataslice0.append(data0[i, :])

                    t0 = process_time()
                    C_old = self.estimate_via_sliding_windows(dataslice0)
                    etimebayes[1] = process_time() - t0

                if k>=1:
                    ##### Bayes approach: Calculate C1 (and any following) using C0 usind discounting
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

            avg_time = sum(etimenaive) / len(etimenaive)
            avg_time_bayes = sum(etimebayes) / len(etimebayes)



            print("Naive", self.nwindow, etimenaive)
            print("Naive Average", avg_time,"\n--------------------")
            print("Bayes", self.nwindow, etimebayes)
            print("Bayes Average", avg_time_bayes, "\n--------------------")





