from unittest import TestCase
from time import process_time
from msmtools.estimation import transition_matrix as _tm
from ldshmm.util.util_functionality import *
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.plottings import ProbPlot
from ldshmm.util.util_math import Utility

class Test_Probability_plot(TestCase):

    def setUp(self):
        self.min_eta = Variable_Holder.min_eta
        self.min_scale_win = Variable_Holder.min_scale_win
        self.min_num_traj = Variable_Holder.min_num_traj
        self.heatmap_size = Variable_Holder.heatmap_size
        self.min_taumeta = Variable_Holder.min_taumeta
        self.taumeta = Variable_Holder.mid_taumeta
        self.mid_eta = Variable_Holder.mid_eta
        self.mid_scale_win = Variable_Holder.mid_scale_win
        self.mid_num_traj = Variable_Holder.mid_num_traj

        self.product_mid_values = Variable_Holder.product_mid_values
        self.numsteps_global = Variable_Holder.numsteps_global


    def test_probability_plot(self):
        plot = ProbPlot()
        rows = 4

        plot.new_plot(rows=rows)
        for k in range(0, 2 * rows):
            self.nstates = 4
            self.mmf1_0 = MMFamily1(self.nstates)
            self.mm1_0_0 = self.mmf1_0.sample()[0]

            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
            self.nstep = Variable_Holder.mid_eta * self.taumeta
            self.nwindow = self.mid_scale_win * self.nstep
            self.numsteps = 128
            self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
            self.ntraj = self.mid_num_traj
            self.r = (self.nwindow - self.nstep) / self.nwindow

            err = np.zeros(self.numsteps + 1, dtype=float)
            errbayes = np.zeros(self.numsteps + 1, dtype=float)

            self.data1_0_0 = []
            for i in range(0, self.ntraj):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.lentraj)))
            dataarray = np.asarray(self.data1_0_0)


            err_bayes = self.performance_and_error_calculation(dataarray, errbayes)
            plot.add_to_plot(err_bayes)

        plot.save_plot("error_probplot")


    def performance_and_error_calculation(self, dataarray, errbayes):
        for k in range(0, self.numsteps + 1):
            data0 = dataarray[:, k * self.nstep: (self.nwindow + k * self.nstep)]
            dataslice0 = []
            for i in range(0, self.ntraj):
                dataslice0.append(data0[i, :])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.nstep: (self.nwindow + 0 * self.nstep)]
                dataslice0 = []
                for i in range(0, self.ntraj):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, nstates=self.nstates)
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.nwindow + (k - 1) * self.nstep - 1: (self.nwindow + k * self.nstep)]
                dataslice1new = []
                for i in range(0, self.ntraj):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     nstates=self.nstates)  # count matrix for just new transitions
                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                errbayes[k] = Utility.log_value(np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans))
        return errbayes


