from time import process_time

from msmtools.estimation import transition_matrix as _tm
import math
from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.spectral_mm import SpectralMM

class Evaluation_Holder():
    """
    Class holding all evaluation functions used for a MMMScaled.
    See util_evaluation.py - The evaluation functions are similar except for the underlying model.
    """

    def error_function (m1, model):
        return np.linalg.norm(m1-model.trans)

    def __init__(self, model, variable_config, error_function = error_function, evaluate_method="both", simulate=True, filename="mm", init_run = False, log_values = True):
        """
        Info: This constructor reads simualted data from a previously created file simulated_data_mm ...

        :param model: MMMScaled (for instance obtained by sampling the MMFamily1)
        :param variable_holder: Variable_Holder objects that holds all configurations
        :param simulate: bool (default=True) - decide whether data should be simulated in the constructor
        :param init_run: bool (default=True)
        """
        if init_run:
            # ToDo Check which data to use for init run
            estimate_via_sliding_windows(data=[], num_states=Variable_Holder.num_states)

        self.model = model
        if simulate:
            self.simulated_data = simulate_and_store(model, filename)
        self.variable_config = variable_config
        self.error_function = error_function
        self.evaluate_method = evaluate_method
        self.log_values = log_values

    def evaluate(self, model=None, simulated_data=None, print_intermediate_values=False):
        if model is not None:
            self.model = model
        if simulated_data is not None:
            self.simulated_data = simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            self.variable_config.heatmap_size)

        for one, values1 in enumerate(self.variable_config.iter_values1):
            for two, values2 in enumerate(self.variable_config.iter_values2):
                if self.variable_config.taumeta is None:
                    self.taumeta = values1
                else:
                    self.tauemta = self.variable_config.taumeta

                if self.variable_config.eta is None:
                    self.eta = values2
                else:
                    self.eta = self.variable_config.eta

                if self.variable_config.scale_window is None:
                    self.scale_window = values2
                else:
                    self.scale_window = self.variable_config.scale_window


                if self.variable_config.num_trajectories is None:
                    self.num_trajectories = values2
                    self.len_trajectory = int (Variable_Holder.num_trajectories_num_transitions_max / self.num_trajectories)+1
                else:
                    self.num_trajectories = self.variable_config.num_trajectories
                    self.len_trajectory = self.variable_config.len_trajectory



                self.model_scaled = self.model.eval(self.taumeta)
                self.shift = self.eta * self.taumeta
                self.window_size = self.scale_window * self.shift


                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory,
                                                                    self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                if print_intermediate_values:
                    self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, self.eta,
                                        self.scale_window)

                dataarray = np.asarray(self.simulated_data)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.performance_and_error_calculation(dataarray)

                if not self.log_values:
                    return log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes, None, None

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes
                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

        return avg_times_naive, avg_errs_naive , avg_times_bayes, avg_errs_bayes, self.variable_config.iter_values1, self.variable_config.iter_values2


    """
    ######
    ### Performance and Error Utility Calculation Methods
    ######
    """

    def performance_and_error_calculation(self, dataarray):
        """
        calculation of both performance and error for naive and bayes

        :param dataarray:
        :return:
        """

        if self.evaluate_method == "bayes":
            time_bayes, error_bayes = self.performance_and_error_calculation_bayes(dataarray)
            if self.log_values:
                time_bayes, error_bayes = self.calc_log_avg_values(time_bayes,error_bayes)
            return  None, None, time_bayes, error_bayes

        elif self.evaluate_method == "naive":
            time_naive, error_naive =  self.performance_and_error_calculation_naive(dataarray)
            if self.log_values:
                time_naive, error_naive = self.calc_log_avg_values(time_naive, error_naive)
            return time_naive, error_naive, None, None

        elif self.evaluate_method == "both":
            time_bayes, error_bayes = self.performance_and_error_calculation_bayes(dataarray)
            time_naive, error_naive = self.performance_and_error_calculation_naive(dataarray)

            if self.log_values:
                time_bayes, error_bayes = self.calc_log_avg_values(time_bayes,error_bayes)

                time_naive, error_naive= self.calc_log_avg_values(time_naive, error_naive)

            return time_naive, error_naive,time_bayes, error_bayes



    def performance_and_error_calculation_naive(self, dataarray):
        """
        performance and error calculation for naive only

        :param dataarray:
        :return:
        """
        lag = int(Variable_Holder.max_taumeta / self.taumeta)
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            t0 = process_time()
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            dataslice0 = convert_2d_to_list_of_rows(data0)

            C0 = estimate_via_sliding_windows(data=dataslice0,
                                              num_states=Variable_Holder.num_states, initial=True, lag=lag)  # count matrix for whole window
            A0 = _tm(C0)
            t1 = process_time()
            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            if type(self.model_scaled) == SpectralMM:
                # stationary
                err[k] = self.error_function(m1=A0, model=self.model_scaled)
            else:
                # non-stationary
                # print(A0)
                # print(self.model_scaled.eval(k).trans)
                dk = int((self.window_size - 1) / 2)
                estimation_time = current_time - dk
                err[k] = self.error_function(m1=A0, model=self.model_scaled.eval(estimation_time))

        return etimenaive, err

    def performance_and_error_calculation_bayes(self, dataarray):
        """
        performance and error calculation for bayes only

        :param dataarray:
        :return:
        """

        lag = int(Variable_Holder.max_taumeta / self.taumeta)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                t0 = process_time()
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = convert_2d_to_list_of_rows(data0)

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True, lag=lag)
                A0 = _tm(C_old)
                etimebayes[1] = process_time() - t0
                if type(self.model_scaled) == SpectralMM:
                    # stationary
                    errbayes[0] = self.error_function(m1=A0, model=self.model_scaled)
                else:
                    # non-stationary
                    #print(A0)
                    #print(self.model_scaled.eval(k).trans)
                    dk = int(self.window_size - (self.shift + 1) / 2 - self.window_size * math.pow(self.r, k + 1) / 2)
                    estimation_time = current_time - dk
                    errbayes[0] = self.error_function(m1=A0, model=self.model_scaled.eval(estimation_time))

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                t0 = process_time()
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = convert_2d_to_list_of_rows(data1new)

                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states, lag=lag)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                if type(self.model_scaled) == SpectralMM:
                    errbayes[k] = self.error_function(m1=A1bayes, model=self.model_scaled)
                else:
                    #print(A1bayes)
                    #print(self.model_scaled.eval(k).trans)
                    dk = int(self.window_size - (self.shift + 1) / 2 - self.window_size * math.pow(self.r, k + 1) / 2)
                    estimation_time = current_time - dk
                    errbayes[k] = self.error_function(m1=A1bayes, model=self.model_scaled.eval(estimation_time))
        return etimebayes, errbayes



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
        print("NAIVE window_size * num_estimations\t", window_size * (num_estimations + 1))
        print("BAYES window_size + num_estimations*shift\t", window_size + num_estimations * shift)
        print("\n")


    def calc_log_avg_values(self, time, error):
        log_total_time = Utility.log_value(time[-1])
        log_avg_err = Utility.log_value(sum(error) / len(error))
        return log_total_time, log_avg_err