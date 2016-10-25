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
        self.num_trajectories = 2
        self.window_size = [int(128*math.pow(2,i)) for i in range (1,7)]
        self.len_trajectory = self.window_size[-1] + 16 * self.shift
        self.num_estimations = [1,2,4,8,16]

    def test_effective_window_size(self):

        for num_estimation in self.num_estimations:
            avg_err_final, avg_err_bayes_final, window_size, effective_window_size_values = self.get_errors(num_estimation)



    def get_errors(self, num_estimations):
        first_run=True
        effective_window_size_values = []
        avg_err, avg_err_bayes = {}, {}
        num_runs = 1

        for j in range(0, num_runs):
            if j > 0:
                first_run = False

            # sample and simulate the trajectory only once for one iteration over the windows values
            self.num_states = 4
            self.mmf1_0 = MMFamily1(self.num_states)
            self.mm1_0_0 = self.mmf1_0.sample()[0]

            self.data1_0_0 = []
            self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)

            for i in range(0, self.num_trajectories):
                self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
            dataarray = np.asarray(self.data1_0_0)
            err_list = []
            err_bayes_list = []

            for window_size in self.window_size:
                self.r = (window_size - self.shift) / window_size
                if first_run:
                    effective_window_size_values.append(self.shift / (1 - self.r))
                err = np.zeros(num_estimations + 1, dtype=float)
                errbayes = np.zeros(num_estimations + 1, dtype=float)
                err, errbayes = self.performance_and_error_calculation(dataarray, err, errbayes, window_size, num_estimations)

                err_list.append(err)
                err_bayes_list.append(errbayes)
            avg_err[j] = err_list
            avg_err_bayes[j] = err_bayes_list
        avg_err_final = np.mean(list(avg_err.values()), axis=0)
        avg_err_bayes_final = np.mean(list(avg_err_bayes.values()), axis=0)

        print(avg_err_final)
        print(avg_err_bayes_final)

        # take the log values
        avg_err_final = [math.log2(x) for x in avg_err_final]
        avg_err_bayes_final = [math.log2(y) for y in avg_err_bayes_final]
        window_size = [math.log2(z) for z in self.window_size]
        effective_window_size_values = [math.log2(a) for a in effective_window_size_values]
        print("Final avg naive errors:", avg_err_final)
        print("Final avg bayes errors:", avg_err_bayes_final)

        return avg_err_final, avg_err_bayes_final, window_size, effective_window_size_values

    def performance_and_error_calculation(self, dataarray, err, errbayes, window_size, num_estimations):
        for k in range(0, num_estimations + 1):
            data0 = dataarray[:, k * self.shift: (window_size + k * self.shift)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # init
                estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)

            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)  # count matrix for whole window
            C0 += 1e-8
            A0 = _tm(C0)
            err[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)
                C_old += 1e-8
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
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
        print("\nNum_estimations: ", num_estimations)
        print("window_size:", window_size)

        print("### Bayes last error ###", errbayes[-1])
        print("### Bayes Error estimation ###", error_estimation_formula(num_estimations, window_size, self.shift, self.r))
        return err[-1], errbayes[-1]

def error_estimation_formula(ne, w, shift, r):
    sum_tmp = 0
    for i in range(0, ne-1):
        sum_tmp+= math.pow(r, i)*math.sqrt(i+1)

    return (math.pow(r, ne) * math.sqrt(w + ne * shift) + math.sqrt(shift) * (1 - r) * sum_tmp) / math.sqrt(w)
