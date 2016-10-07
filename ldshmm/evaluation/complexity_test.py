from unittest import TestCase
import numpy as np
from msmtools.estimation import transition_matrix as _tm
from time import process_time
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.util_math import Utility
from ldshmm.util.util_functionality import *

class Complexity_Test(TestCase):
    def setUp(self):
        self.num_states = 4
        self.mmf1_0 = MMFamily1(self.num_states)
        self.mm1_0_0 = self.mmf1_0.sample()[0]
        self.num_estimations = 100
        self.num_trajectories = 20

    def test_complexity(self):
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)

        # specify values for taumeta to iterate over - taumeta influences shifts and therefore the datacliece size
        taumeta_values = [2,4,8]

        for one,taumeta in enumerate(taumeta_values):
            # Setting taumeta and eta values and recalculate dependent variables for scaling
            self.taumeta = taumeta
            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
            self.shift = 100 * self.taumeta
            self.nwindow = 10 * self.shift
            self.num_estimations = 100
            self.len_trajectory = self.nwindow + self.num_estimations * self.shift + 1
            self.num_trajectories = 20
            self.r = (self.nwindow - self.shift) / self.nwindow

            self.data1_0_0 = []
            for i in range(0, self.num_trajectories):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(self.len_trajectory))
            dataarray = np.asarray(self.data1_0_0)

            for k in range(0, self.num_estimations + 1):
                ##### naive sliding window approach
                data0 = dataarray[:, k * self.shift: (self.nwindow + k * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])
                t0 = process_time()
                C0 = estimate_via_sliding_windows(dataslice0, num_states=self.num_states)  # count matrix for whole window
                t1 = process_time()
                A0 = _tm(C0)
                etimenaive[k + 1] = t1 - t0 + etimenaive[k]


                if k==0:
                    ##### Bayes approach: Calculate C0 separately
                    data0 = dataarray[:, 0 * self.shift: (self.nwindow + 0 * self.shift)]
                    dataslice0 = []
                    for i in range(0, self.num_trajectories):
                        dataslice0.append(data0[i, :])

                    t0 = process_time()
                    C_old = estimate_via_sliding_windows(dataslice0, num_states=self.num_states)
                    etimebayes[1] = process_time() - t0

                if k>=1:
                    ##### Bayes approach: Calculate C1 (and any following) using C0 usind discounting
                    data1new = dataarray[:, self.nwindow + (k - 1) * self.shift - 1: (self.nwindow + k * self.shift)]
                    dataslice1new = []
                    for i in range(0, self.num_trajectories):
                        dataslice1new.append(data1new[i, :])
                    t0 = process_time()
                    C_new = estimate_via_sliding_windows(dataslice1new, num_states=self.num_states)  # count matrix for just new transitions

                    weight0 = self.r
                    weight1 = 1.0

                    C1bayes = weight0 * C_old + weight1 * C_new
                    C_old = C1bayes

                    t1 = process_time()
                    etimebayes[k + 1] = t1 - t0 + etimebayes[k]

            # avg_time = sum(etimenaive)/len(etimenaive)
            avg_time = Utility.calc_slope(etimenaive)
            # avg_time_bayes = sum(etimebayes)/len(etimebayes)
            avg_time_bayes = Utility.calc_slope(etimebayes)



            print("Naive", self.nwindow, etimenaive)
            print("Naive Slope", avg_time,"\n--------------------")
            print("Bayes", self.nwindow, etimebayes)
            print("Bayes Slope", avg_time_bayes, "\n--------------------")





