from unittest import TestCase
import math
import numpy as np
import pyemma.msm as MSM
import pyemma.msm.estimators as _MME
from msmtools.estimation import transition_matrix as _tm
from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from msmtools.estimation.dense.tmatrix_sampler import TransitionMatrixSampler as TMS
from time import process_time

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1

class Approach_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.timescaledisp = 2.0
        self.statconc = 1.0
        self.mmf1_0 = MMFamily1(self.nstates, self.timescaledisp, self.statconc)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0)
        self.qmm1_0_0 = self.qmmf1_0.sample()[0]

        self.taumeta = 3
        self.tauquasi = self.timescaledisp * 3
        self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta, self.tauquasi)

        self.nu = 10
        self.nstep = math.ceil(self.nu * self.timescaledisp * self.taumeta * self.tauquasi)
        self.nwindow = 10 * self.nstep
        self.numsteps = 10
        self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
        self.ntraj = 10
        self.r = (self.nwindow - self.nstep) / self.nwindow

        self.data1_0_0 = []
        for i in range(0, self.ntraj):
            self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(self.lentraj))

    def estimate_via_sliding_windows(self, data):
        C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates = self.nstates)
        return C

    def test_approach2(self):

        dataarray = np.asarray(self.data1_0_0)
        etime = np.zeros(self.numsteps + 2, dtype=float)
        err = np.zeros(self.numsteps + 1, dtype=float)
        etime[0] = 0
        for k in range(0, self.numsteps+1):
            data0 = dataarray[:, k * self.nstep: (self.nwindow + k * self.nstep)]
            dataslice0 = []
            for i in range(0, self.ntraj):
                dataslice0.append(data0[i, :])
            t0 = process_time()
            C0 = self.estimate_via_sliding_windows(dataslice0)  # count matrix for whole window
            t1 = process_time()
            A0 = _tm(C0)
            etime[k+1] = t1-t0 + etime[k]
            err[k] = np.linalg.norm(A0 - self.qmm1_0_0_scaled.eval(self.nwindow + (k - 0.5) * self.nstep).trans)
        print("Times (Windows): ", etime)
        print("Errors (Windows): ", err)

        print("\n############## Bayes #############")
        etimebayes = np.zeros(self.numsteps + 2, dtype=float)
        errbayes = np.zeros(self.numsteps + 1, dtype=float)

        weight0 = self.r
        weight1 = 1.0

        data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
        dataslice0 = []
        for i in range(0, self.ntraj):
            dataslice0.append(data0[i, :])

        t0 = process_time()
        C_old = self.estimate_via_sliding_windows(dataslice0)
        etimebayes[1] = process_time() - t0
        k=0
        errbayes[0] = np.linalg.norm(_tm(C_old) - self.qmm1_0_0_scaled.eval(self.nwindow + (k - 0.5) * self.nstep).trans)

        for k in range(1,self.numsteps+1):
            data1new = dataarray[:, self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
            dataslice1new = []
            for i in range(0, self.ntraj):
                dataslice1new.append(data1new[i, :])
            t0 = process_time()
            C_new = self.estimate_via_sliding_windows(dataslice1new)  # count matrix for just new transitions

            C1bayes = weight0 * C_old + weight1 * C_new
            C_old = C1bayes
            t1 = process_time()
            etimebayes[k+1] = t1 - t0 + etimebayes[k]
            A1bayes = _tm(C1bayes)
            errbayes[k] = np.linalg.norm(A1bayes - self.qmm1_0_0_scaled.eval(self.nwindow + (k - 0.5) * self.nstep).trans)

            tms = TMS(C1bayes)
            nbsamples = 100
            postsample = tms.sample(nbsamples)
            avgerrbayes = 0
            # varbayes = 0
            for i in range(0, nbsamples):
                avgerrbayes = avgerrbayes + np.linalg.norm(A1bayes - postsample[i])/nbsamples
                # varbayes = varbayes + pow(np.linalg.norm(A1bayes - postsample[i]), 2)
            # stdbayes = pow(varbayes, 0.5)
        print("Times (Bayes): ", etimebayes)
        print("Errors (Bayes): ", errbayes)
        print("Posterior Error: ", avgerrbayes)
        #print("Posterior Standard Deviation: ", stdbayes)
