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
        self.min_scale_window = Variable_Holder.min_scale_window
        self.min_num_traj = Variable_Holder.min_num_trajectories
        self.heatmap_size = Variable_Holder.heatmap_size
        self.min_taumeta = Variable_Holder.min_taumeta
        self.taumeta = Variable_Holder.mid_taumeta
        self.mid_eta = Variable_Holder.mid_eta
        self.mid_scale_window = 512#Variable_Holder.mid_scale_window
        self.mid_num_trajectories = Variable_Holder.mid_num_trajectories
        self.mid_taumeta = Variable_Holder.mid_taumeta

        self.shift_mid = self.mid_eta * self.mid_taumeta
        self.window_size_mid = self.mid_scale_window * self.shift_mid
        self.num_estimations_mid = Utility.calc_num_estimations_mid(self.window_size_mid, self.heatmap_size, self.shift_mid)
        self.len_trajectory = int(self.window_size_mid + self.num_estimations_mid * self.shift_mid + 1)

    def test_probability_plot(self):
        plot = ProbPlot()
        rows = 4

        plot.new_plot(rows=rows)
        for k in range(0, 2 * rows):
            self.nstates = 4
            self.mmf1_0 = MMFamily1(self.nstates)
            self.mm1_0_0 = self.mmf1_0.sample()[0]

            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
            self.shift = Variable_Holder.mid_eta * self.taumeta
            self.window_size = self.mid_scale_window * self.shift
            self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
            self.num_trajectories = self.mid_num_trajectories
            self.r = (self.window_size - self.shift) / self.window_size

            errbayes = np.zeros(self.num_estimations + 1, dtype=float)

            self.data1_0_0 = []
            for i in range(0, self.num_trajectories):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
            dataarray = np.asarray(self.data1_0_0)


            err_bayes = self.performance_and_error_calculation(dataarray, errbayes)
            plot.add_to_plot(err_bayes)

        plot.save_plot("error_probplot")


    def performance_and_error_calculation(self, dataarray, errbayes):
        for k in range(0, self.num_estimations + 1):
            data0 = dataarray[:, k * self.shift: (self.window_size + k * self.shift)]
            dataslice0 = []
            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.nstates)
                errbayes[0] = Utility.log_value(np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans))

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (self.window_size + k * self.shift)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=self.nstates)  # count matrix for just new transitions
                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                errbayes[k] = Utility.log_value(np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans))
        return errbayes


