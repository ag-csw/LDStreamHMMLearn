from time import process_time

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.mm_family import MMFamily1

class Evaluation_Holder_MM():
    """
   Class holding all evaluation functions used for a MMMScaled.
   See util_evaluation.py - The evaluation functions are similar except for the underlying model.
   """

    def __init__(self, mm1_0_0, simulate=True, filename="mm", init_run = False):
        """
        Info: This constructor reads simualted data from a previously created file simulated_data_mm ...

        :param mm1_0_0: MMMScaled (for instance obtained by sampling the MMFamily1)
        :param simulate: bool (default=True) - decide whether data should be simulated in the constructor
        :param init_run: bool (default=True)
        """
        if init_run:
            # ToDo Check which data to use for init run
            estimate_via_sliding_windows(data=[], num_states=Variable_Holder.num_states)

        self.mm1_0_0 = mm1_0_0
        if simulate:
            simulate_and_store_data(mm1_0_0, filename)
        self.simulated_data = read_simulated_data(filename)

    def test_taumeta_eta(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                # ToDo Document The formulas used here need justification
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, eta, Variable_Holder.mid_scale_window)

                log_avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_errs_bayes, taumeta_values, eta_values

    def test_taumeta_eta_performance_only(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                # ToDo Document The formulas used here need justification
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, eta,
                                        Variable_Holder.mid_scale_window)

                log_total_time_bayes = self.helper_performance_only(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_bayes[two][one] = log_total_time_bayes

        return avg_times_bayes, taumeta_values, eta_values

    def helper(self, len_trajectory, num_trajectories):
        self.data1_0_0 = []
        dataarray = np.asarray(self.simulated_data[self.taumeta])
        dataarray = dataarray[:num_trajectories]
        dataarray = np.asarray([ndarr[:len_trajectory] for ndarr in dataarray])
        print(dataarray)
        try:
            return self.error_bayes(dataarray)
        except Exception as e:
            print("Exception thrown:", e)
            return self.helper(len_trajectory, num_trajectories)


    def helper_performance_only(self, len_trajectory, num_trajectories):
        self.data1_0_0 = []
        dataarray = np.asarray(self.simulated_data[self.taumeta])
        dataarray = dataarray[:num_trajectories]
        dataarray = np.asarray([ndarr[:len_trajectory] for ndarr in dataarray])
        print(dataarray)
        try:
            return self.performance_bayes(dataarray)
        except Exception as e:
            print("Exception thrown:", e)
            return self.helper_performance_only(len_trajectory, num_trajectories)

    def test_taumeta_scale_window_performance_only(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data = simulated_data

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size,
                                                                    self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, Variable_Holder.len_trajectory, self.num_trajectories,
                                        Variable_Holder.mid_eta,
                                        scale_window)

                log_total_time_bayes = self.helper_performance_only(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_bayes[two][one] = log_total_time_bayes

        return avg_times_bayes, taumeta_values, scale_window_values

    def test_taumeta_num_traj_performance_only(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(
            Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.window_size = (Variable_Holder.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(Variable_Holder.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta,
                                        Variable_Holder.mid_scale_window)

                log_total_time_bayes = self.helper_performance_only(self.len_trajectory, self.num_trajectories)

                avg_times_bayes[two][one] = log_total_time_bayes

        return avg_times_bayes, taumeta_values, num_traj_values


    def test_taumeta_scale_window(self, mm1_0_0, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta,
                                        scale_window)

                log_avg_err_bayes = self.helper(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_errs_bayes, taumeta_values, scale_window_values

    def test_taumeta_num_traj(self, mm1_0_0=None, simulated_data=None):
        if mm1_0_0:
            self.mm1_0_0 = mm1_0_0
        if simulated_data:
            self.simulated_data=simulated_data
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.window_size = (Variable_Holder.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(Variable_Holder.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, Variable_Holder.mid_scale_window)

                log_avg_err_bayes = self.helper(self.len_trajectory, self.num_trajectories)

                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_errs_bayes, taumeta_values, num_traj_values


    def test_timescaledisp_eta(self):

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        timescaledisp_values = create_value_list(Variable_Holder.min_timescaledisp, Variable_Holder.heatmap_size)
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, timescaledisp in enumerate(timescaledisp_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                # ToDo Document The formulas used here need justification
                self.timescaledisp = timescaledisp
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=self.timescaledisp,
                                        statconc=Variable_Holder.mid_statconc)
                self.mm1_0_0 = self.mmf1_0.sample()[0]

                self.taumeta = Variable_Holder.mid_taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, eta, Variable_Holder.mid_scale_window)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.helper_timescaledisp(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, timescaledisp_values, eta_values

    def helper_timescaledisp(self, len_trajectory, num_trajectories):
        self.data1_0_0 = []
        dataarray = np.asarray(self.simulated_data[self.timescaledisp])
        dataarray = dataarray[:num_trajectories]
        dataarray = np.asarray([ndarr[:len_trajectory] for ndarr in dataarray])
        print(dataarray)
        try:
            return self.performance_and_error_calculation(dataarray
              )
        except Exception as e:
            print("Exception thrown:", e)
            return self.helper(len_trajectory, num_trajectories)

    def test_timescaledisp_scale_window(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        timescaledisp_values = create_value_list(Variable_Holder.min_timescaledisp, Variable_Holder.heatmap_size)
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, timescaledisp in enumerate(timescaledisp_values):
            for two, scale_window in enumerate(scale_window_values):
                self.timescaledisp = timescaledisp
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=self.timescaledisp,
                                        statconc=Variable_Holder.mid_statconc)
                self.mm1_0_0 = self.mmf1_0.sample()[0]

                self.taumeta = Variable_Holder.mid_taumeta

                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta,
                                        scale_window)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.helper_timescaledisp(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, timescaledisp_values, scale_window_values

    def test_timescaledisp_num_traj(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        timescaledisp_values = create_value_list(Variable_Holder.min_timescaledisp, Variable_Holder.heatmap_size)
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, timescaledisp in enumerate(timescaledisp_values):
            for two, num_traj in enumerate(num_traj_values):
                self.timescaledisp = timescaledisp
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=self.timescaledisp,
                                        statconc=Variable_Holder.mid_statconc)
                self.mm1_0_0 = self.mmf1_0.sample()[0]

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = Variable_Holder.mid_taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.window_size = (Variable_Holder.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(Variable_Holder.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, Variable_Holder.mid_scale_window)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.helper_timescaledisp(self.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, timescaledisp_values, num_traj_values


    def test_statconc_eta(self):

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        statconc_values = Variable_Holder.statconc_values
        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)

        for one, statconc in enumerate(statconc_values):
            for two, eta in enumerate(eta_values):
                # Setting taumeta and eta values and recalculate dependent variables for scaling
                # ToDo Document The formulas used here need justification
                self.statconc = statconc
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=Variable_Holder.mid_timescaledisp,
                                        statconc=self.statconc)
                self.mm1_0_0 = self.mmf1_0.sample()[0]

                self.taumeta = Variable_Holder.mid_taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        Variable_Holder.len_trajectory, self.num_trajectories, eta, Variable_Holder.mid_scale_window)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.helper_statconc(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, statconc_values, eta_values

    def helper_statconc(self, len_trajectory, num_trajectories):
        self.data1_0_0 = []
        dataarray = np.asarray(self.simulated_data[self.statconc])
        dataarray = dataarray[:num_trajectories]
        dataarray = np.asarray([ndarr[:len_trajectory] for ndarr in dataarray])
        print(dataarray)
        try:
            return self.performance_and_error_calculation(dataarray)
        except Exception as e:
            print("Exception thrown:", e)
            return self.helper(len_trajectory, num_trajectories)

    def test_statconc_scale_window(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        statconc_values = Variable_Holder.statconc_values
        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)

        for one, statconc in enumerate(statconc_values):
            for two, scale_window in enumerate(scale_window_values):
                self.statconc = statconc
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=Variable_Holder.mid_timescaledisp,
                                        statconc=self.statconc)
                self.mm1_0_0 = self.mmf1_0.sample()[0]

                self.taumeta = Variable_Holder.mid_taumeta

                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # ToDo Document Some of these formulas are based on essential definitions
                # e.g. scale_window is defined to be self.shift/self.window_size
                self.window_size = scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories
                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size,
                                        self.num_estimations, Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta,
                                        scale_window)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.helper_statconc(Variable_Holder.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, statconc_values, scale_window_values

    def test_statconc_num_traj(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(Variable_Holder.heatmap_size)

        # specify values for taumeta and eta to iterate over
        statconc_values = Variable_Holder.statconc_values
        num_traj_values = create_value_list(Variable_Holder.min_num_trajectories, Variable_Holder.heatmap_size)

        for one, statconc in enumerate(statconc_values):
            for two, num_traj in enumerate(num_traj_values):
                self.statconc= statconc
                self.mmf1_0 = MMFamily1(Variable_Holder.num_states, timescaledisp=Variable_Holder.mid_timescaledisp,
                                        statconc=self.statconc)
                self.mm1_0_0 = self.mmf1_0.sample()[0]

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = Variable_Holder.mid_taumeta
                self.mm1_0_0_scaled = self.mm1_0_0.eval(self.taumeta)
                self.shift = (Variable_Holder.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.window_size = (Variable_Holder.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(Variable_Holder.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ", self.taumeta, self.shift, self.window_size, self.num_estimations,
                                        self.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, Variable_Holder.mid_scale_window)

                log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes = self.helper_statconc(self.len_trajectory, self.num_trajectories)

                avg_times_naive[two][one] = log_total_time_naive
                avg_errs_naive[two][one] = log_avg_err_naive

                avg_times_bayes[two][one] = log_total_time_bayes
                avg_errs_bayes[two][one] = log_avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, statconc_values, num_traj_values



    def performance_and_error_calculation(self, dataarray):
        #ToDo Document

        log_total_time_naive, log_avg_err_naive = self.performance_and_error_calculation_naive(dataarray)
        log_total_time_bayes, log_avg_err_bayes = self.performance_and_error_calculation_bayes(dataarray)

        return log_total_time_naive, log_avg_err_naive, log_total_time_bayes, log_avg_err_bayes


    def performance_and_error_calculation_naive(self, dataarray):
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            t0 = process_time()
            data0 = dataarray[:, current_time - self.window_size + 1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])

            C0 = estimate_via_sliding_windows(data=dataslice0,
                                              num_states=Variable_Holder.num_states, initial=True)  # count matrix for whole window
            A0 = _tm(C0)
            t1 = process_time()

            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            err[k] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)

        log_total_time_naive = Utility.log_value(etimenaive[-1])
        log_avg_err_naive = Utility.log_value(sum(err) / len(err))

        return log_total_time_naive, log_avg_err_naive



    def performance_and_error_calculation_bayes(self, dataarray):
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                t0 = process_time()
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True)
                A0 = _tm(C_old)
                etimebayes[1] = process_time() - t0
                errbayes[0] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                t0 = process_time()
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)

        log_total_time_bayes = Utility.log_value(etimebayes[-1])
        log_avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))
        return log_total_time_bayes, log_avg_err_bayes


    def performance_bayes(self, dataarray):
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                t0 = process_time()
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True)
                A0 = _tm(C_old)
                etimebayes[1] = process_time() - t0

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                t0 = process_time()
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]

        log_total_time_bayes = Utility.log_value(etimebayes[-1])
        return log_total_time_bayes


    def error_bayes(self, dataarray):
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift - 1
            assert (current_time < np.shape(dataarray)[1])
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=Variable_Holder.num_states, initial=True)
                A0 = _tm(C_old)
                errbayes[0] = np.linalg.norm(A0 - self.mm1_0_0_scaled.trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                C_new = estimate_via_sliding_windows(data=dataslice1new,
                                                     num_states=Variable_Holder.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans)

        log_avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))
        return log_avg_err_bayes



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