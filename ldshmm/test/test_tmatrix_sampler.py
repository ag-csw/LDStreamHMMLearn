from unittest import TestCase
from msmtools.estimation.dense.tmatrix_sampler import TransitionMatrixSampler
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.mm_family import MMFamily1
import numpy as np
from time import process_time
from msmtools.estimation import transition_matrix as _tm
from ldshmm.util.util_functionality import estimate_via_sliding_windows



class Test_TMatrix_Sampler(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]


    def test_create(self):

        # Setting taumeta and eta values and recalculate dependent variables for scaling
        self.taumeta = Variable_Holder.min_taumeta
        self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
        self.nstep = Variable_Holder.mid_eta * self.taumeta
        self.nwindow = Variable_Holder.mid_scale_win * self.nstep
        self.numsteps = int(Variable_Holder.numsteps_global / Variable_Holder.product_mid_values)
        self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
        self.ntraj = Variable_Holder.mid_num_traj
        self.r = (self.nwindow - self.nstep) / self.nwindow

        etimebayes = np.zeros(self.numsteps + 2, dtype=float)
        errbayes = np.zeros(self.numsteps + 1, dtype=float)

        self.data1_0_0 = []
        for i in range(0, self.ntraj):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.lentraj)))
        dataarray = np.asarray(self.data1_0_0)

        for k in range(0, 2):

            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
                dataslice0 = []
                for i in range(0, self.ntraj):
                    dataslice0.append(data0[i, :])
                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, nstates=self.nstates)
                etimebayes[1] = process_time() - t0
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
                dataslice1new = []
                for i in range(0, self.ntraj):
                    dataslice1new.append(data1new[i, :])
                t0 = process_time()
                C_new = estimate_via_sliding_windows(data=dataslice1new, nstates=self.nstates)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes
                print("Count Matrix:", C1bayes)
                tmatrix_sampler = TransitionMatrixSampler(C=C1bayes)
                samples = tmatrix_sampler.sample(nsamples=1)
                print(samples)

                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)

