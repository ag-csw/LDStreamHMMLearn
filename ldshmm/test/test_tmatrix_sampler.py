from unittest import TestCase
from msmtools.estimation.dense.tmatrix_sampler import TransitionMatrixSampler
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.mm_family import MMFamily1
import numpy as np
from time import process_time
from msmtools.estimation import transition_matrix as _tm
from ldshmm.util.util_functionality import estimate_via_sliding_windows
from ldshmm.util.plottings import PointPlot



class Test_TMatrix_Sampler(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]


    def test_create(self):

        plt = PointPlot()
        plt.new_plot("Error Comparison", rows=4)

        for l in range(0,8):
            # Setting taumeta and eta values and recalculate dependent variables for scaling
            self.taumeta = Variable_Holder.min_taumeta
            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
            self.shift = Variable_Holder.mid_eta * self.taumeta
            self.window_size = Variable_Holder.mid_scale_window * self.shift
            self.num_estimations = int(Variable_Holder.num_estimations_global / Variable_Holder.product_mid_values)
            self.len_trajectory = self.window_size + self.num_estimations * self.shift + 1
            self.num_trajectories = 16#Variable_Holder.mid_num_traj
            self.r = (self.window_size - self.shift) / self.window_size

            errbayes = np.zeros(self.num_estimations, dtype=float)
            errTMatrSampl = np.zeros(self.num_estimations, dtype=float)

            self.data1_0_0 = []
            for i in range(0, self.num_trajectories):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
            dataarray = np.asarray(self.data1_0_0)

            for k in range(0, self.num_estimations):
                print(k)
                if k == 0:
                    ##### Bayes approach: Calculate C0 separately
                    data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                    dataslice0 = []
                    for i in range(0, self.num_trajectories):
                        dataslice0.append(data0[i, :])
                    C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.nstates)
                    errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

                    tmatrix_sampler = TransitionMatrixSampler(C=C_old)
                    samples = tmatrix_sampler.sample(nsamples=128)
                    average = np.mean(samples, axis=0)
                    err = np.zeros(shape=np.shape(samples))

                    for l, sample in enumerate(samples):
                        diff = sample - average
                        err[l] = np.linalg.norm(diff)

                    average_err = np.mean(err)

                    errTMatrSampl[k] = average_err

                if k >= 1:
                    ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                    data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (self.window_size + k * self.shift)]
                    dataslice1new = []
                    for i in range(0, self.num_trajectories):
                        dataslice1new.append(data1new[i, :])
                    C_new = estimate_via_sliding_windows(data=dataslice1new, num_states=self.nstates)  # count matrix for just new transitions

                    weight0 = self.r
                    weight1 = 1.0

                    C1bayes = weight0 * C_old + weight1 * C_new
                    C_old = C1bayes

                    t1 = process_time()
                    A1bayes = _tm(C1bayes)
                    errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)

                    tmatrix_sampler = TransitionMatrixSampler(C=C1bayes)
                    samples = tmatrix_sampler.sample(nsamples=128)
                    average = np.mean(samples, axis=0)
                    err = np.zeros(shape=np.shape(samples))

                    for l, sample in enumerate(samples):
                        diff = sample - average
                        err[l] = np.linalg.norm(diff)

                    average_err = np.mean(err)
                    errTMatrSampl[k] = average_err

            plt.add_to_plot(err_data=errbayes, tmatrix_err_data=errTMatrSampl)
        plt.save_plot("error_comparison")

