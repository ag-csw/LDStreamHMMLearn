from unittest import TestCase
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.util_functionality import *
import math
from msmtools.estimation import transition_matrix as _tm
from ldshmm.util.plottings import PointPlot

class Effective_Window_Size_Test(TestCase):
    def setUp(self):

        self.taumeta = 4
        self.shift = 64
        self.num_trajectories = 1
        self.window_size = [int(128*math.pow(2,i)) for i in range (1,7)]
        self.len_trajectory = self.window_size[-1] + 16 * self.shift
        self.num_estimations=16

    def test_effective_window_size(self):
        avg_err_bayes, window_size = self.get_errors(self.num_estimations)

        naive = avg_err_bayes[0]

        plot = PointPlot()
        plot.new_plot("Effective Window Size", rows=1)
        plot.new_subplot()

        for value in avg_err_bayes:
            plot.add_data_to_plot(value, window_size, value)
        plot.new_subplot()

        for value in naive:
            plot.add_data_to_plot(value, window_size, value)

        plot.create_legend()
        plot.save_plot("effective_window_size_plot")

    def get_errors(self, num_estimations):
        err_bayes_dict = {}
        num_runs = 256

        for j in range(0, num_runs):

            # sample and simulate the trajectory only once for one iteration over the windows values
            self.num_states = 4
            self.mmf1_0 = MMFamily1(self.num_states)
            self.mm1_0_0 = self.mmf1_0.sample()[0]

            self.data1_0_0 = []
            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)

            for i in range(0, self.num_trajectories):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
            dataarray = np.asarray(self.data1_0_0)
            err_bayes_list = []

            for window_size in self.window_size:
                self.r = (window_size - self.shift) / window_size

                errbayes = self.performance_and_error_calculation(dataarray, window_size, num_estimations)

                err_bayes_list.append(errbayes)
            err_bayes_dict[j] = err_bayes_list
        avg_err_bayes = np.mean(list(err_bayes_dict.values()), axis=0)

        # take the log values
        avg_err_bayes = [math.log2(y) for y in avg_err_bayes]
        window_size = [math.log2(z) for z in self.window_size]
        print("Avg bayes errors:", avg_err_bayes)

        return avg_err_bayes, window_size

    def performance_and_error_calculation(self, dataarray, window_size, num_estimations):
        errbayes = np.zeros(num_estimations + 1, dtype=float)
        for k in range(0, num_estimations + 1):
            data0 = dataarray[:, k * self.shift: (window_size + k * self.shift)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])


            if k == 0:
                data0 = dataarray[:, 0 * self.shift: (window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)
                C_old += 1e-8
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)
            if k >= 1:
                data1new = dataarray[:, window_size + (k - 1) * self.shift - 1: (window_size + k * self.shift)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=self.num_states)  # count matrix for just new transitions
                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes
                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)
            self.print_values(window_size, num_estimations, errbayes[k], errbayes[0])
        return errbayes


    def print_values(self, window_size, num_estimations, bayes_error, naive_error):
        print("**********")
        print("Window Size:", window_size)
        print("Number of Estimations:", num_estimations)
        print("Shift:",self.shift)
        print("Actual Expected Bayes/Naive Ratio:", bayes_error/naive_error)
        print("Theoretical Bound for Expected Bayes/Naive Ratio:", self.error_estimation_formula((num_estimations,window_size,self.shift,self.r)))


    def error_estimation_formula(self, ne, w, shift, r):
        sum_tmp = 0
        for i in range(0, ne-1):
            sum_tmp+= math.pow(r, i)*math.sqrt(i+1)

        return (math.pow(r, ne) * math.sqrt(w + ne * shift) + math.sqrt(shift) * (1 - r) * sum_tmp) / math.sqrt(w)
