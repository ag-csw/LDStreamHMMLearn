from time import process_time
from unittest import TestCase

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder


class New_Naive_Test(TestCase):
    def setUp(self):

        self.num_states = 4
        self.mmf1_0 = MMFamily1(self.num_states)
        self.mm1_0_0 = self.mmf1_0.sample()[0]

        self.min_eta=Variable_Holder.min_eta
        self.min_scale_window=Variable_Holder.min_scale_window
        self.min_num_trajectories=Variable_Holder.min_num_trajectories
        self.heatmap_size=Variable_Holder.heatmap_size
        self.min_taumeta=Variable_Holder.min_taumeta

        self.mid_eta = Variable_Holder.mid_eta
        self.mid_scale_window = Variable_Holder.mid_scale_window
        self.mid_num_trajectories = Variable_Holder.mid_num_trajectories
        self.mid_taumeta = Variable_Holder.mid_taumeta

        self.max_eta = Variable_Holder.max_eta
        self.max_taumeta = Variable_Holder.max_taumeta
        self.shift_max = Variable_Holder.shift_max
        self.window_size_max = Variable_Holder.window_size_max
        self.num_estimations_max = Variable_Holder.window_size_max

        self.num_trajectories_max = Variable_Holder.num_trajectories_max

        self.len_trajectory = Variable_Holder.len_trajectory
        self.num_trajectories_len_trajectory_max = Variable_Holder.num_trajectories_len_trajectory_max

    def test_run_all_tests(self):
        avg_errs_naive1_list =  {}
        avg_errs_bayes1_list = {}
        avg_errs_naive_new_list = {}

        taumeta_values = []
        eta_values = []

        for i in range (0,8):
            # calculate performances and errors for the three parameters
            avg_errs_naive, avg_errs_naive_new, avg_errs_bayes, taumeta_values, eta_values = self.test_taumeta_eta()
            avg_errs_naive1_list[i] = (avg_errs_naive)
            avg_errs_naive_new_list[i] = avg_errs_naive_new
            avg_errs_bayes1_list[i] = avg_errs_bayes


        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=1, cols=3)

        avg_errs_naive = np.mean(list(avg_errs_naive1_list.values()), axis=0)
        avg_errs_naive_new = np.mean(list(avg_errs_naive_new_list.values()), axis=0)
        avg_errs_bayes = np.mean(list(avg_errs_bayes1_list.values()), axis=0)

        print("NAIVE", avg_errs_naive)
        print("NAIVE NEW", avg_errs_naive_new)
        print("BAYES", avg_errs_bayes)

        # get minimum and maximum error
        min_val = np.amin([avg_errs_naive, avg_errs_naive_new, avg_errs_bayes])
        max_val = np.amax([avg_errs_naive, avg_errs_naive_new, avg_errs_bayes])

        # input data into one plot
        plots.add_to_plot_same_colorbar_new(data_naive=avg_errs_naive, data_naive_new = avg_errs_naive_new, data_bayes=avg_errs_bayes, x_labels=taumeta_values,
                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)


        plots.save_plot_same_colorbar("Error")

    def test_taumeta_eta(self):
        avg_errs_naive, avg_errs_naive_new, avg_errs_bayes= init_error_arrays_new(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        eta_values = create_value_list(self.min_eta, self.heatmap_size)
        first=True
        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                self.window_size = self.mid_scale_window * self.shift
                ######################### in the iterations, except the first run, we add the shift to the window size
                if first:
                    first=False
                else:
                    self.window_size += self.shift
                #########################
                self.num_trajectories = self.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, eta, self.mid_scale_window)

                avg_err_naive, avg_err_naive_new, avg_err_bayes = self.test_eta_helper()

                avg_errs_naive[two][one] = avg_err_naive
                avg_errs_naive_new[two][one] = avg_err_naive_new
                avg_errs_bayes[two][one] = avg_err_bayes


        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""

        return avg_errs_naive, avg_errs_naive_new, avg_errs_bayes, taumeta_values, eta_values


    def test_eta_helper(self):
        err_naive = np.zeros(self.num_estimations + 1, dtype=float)
        err_naive_new = np.zeros(self.num_estimations + 1, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)

        ##################### Data for the new naive approach
        data_new = []
        for i in range(0, self.num_trajectories):
            data_new.append(self.mm1_0_0_scaled.simulate(int(self.len_trajectory+self.shift)))
        dataarray_new = np.asarray(data_new)
        #####################
        try:
            return self.performance_and_error_calculation(
                dataarray, dataarray_new, err_naive, err_naive_new, errbayes)
        except:
            return self.test_eta_helper()

    def performance_and_error_calculation(self, dataarray, dataarray_new, err_naive, err_naive_new, errbayes):

        for k in range(0, self.num_estimations + 1):
            assert (self.window_size + k * self.shift) < np.shape(dataarray)[1]
            data0 = dataarray[:, k * self.shift: (self.window_size + k * self.shift)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k == 0:
                # init
                estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)
            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)  # count matrix for whole window
            C0 += 1e-8
            A0 = _tm(C0)
            err_naive[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)

            ############## New naive approach:
            data0_new = dataarray_new[:, k * self.shift: (self.window_size + k * self.shift)]
            dataslice0_new = []

            for i in range(0, self.num_trajectories):
                dataslice0_new.append(data0_new[i, :])
            C0 = estimate_via_sliding_windows(data=dataslice0_new,
                                              num_states=self.num_states)  # count matrix for whole window
            C0 += 1e-8
            A0 = _tm(C0)
            err_naive_new[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)
            ##############

            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)
                # f.write("BAYES "+str(etimebayes[1])+"\n")
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (self.window_size + k * self.shift)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                t0 = process_time()
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=self.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                t1 = process_time()
                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)
        avg_err_naive = Utility.log_value(sum(err_naive) / len(err_naive))
        avg_err_naive_new = Utility.log_value(sum(err_naive_new) / len(err_naive_new))
        avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))

        return avg_err_naive, avg_err_naive_new, avg_err_bayes


    def print_param_values(self, evaluation_name, taumeta, shift, window_size, num_estimations, len_trajectory,
                           num_trajectories, eta, scale_window):
        print("Parameter Overview for " + evaluation_name + ":")
        print("taumeta:\t", taumeta)
        print("eta:\t", eta)
        print("scale_window\t:", scale_window)
        print("shift:\t", shift)
        print("window_size:\t", window_size)
        print("num_estimations:\t", num_estimations)
        print("len_trajectory:\t", len_trajectory)
        print("num_trajectories:\t", num_trajectories)
        print("num_trajectories*len_trajectory:\t", num_trajectories * len_trajectory)
        #print("NAIVE window_size * num_estimations\t", window_size * (num_estimations + 1))
        #print("BAYES window_size + num_estimations*shift\t", window_size + num_estimations * shift)
        print("lentraj + shift:\t", len_trajectory+shift)
        print("\n")
