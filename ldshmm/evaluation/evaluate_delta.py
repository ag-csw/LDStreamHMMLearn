from time import process_time

from msmtools.estimation import transition_matrix as _tm

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_functionality import *
from ldshmm.util.util_math import Utility
from ldshmm.util.variable_holder import Variable_Holder


class Delta_Evaluation():

    def __init__(self, delta=0):
        self.num_states = 4
        self.delta = delta

        np.random.seed(1)
        self.mmf1_0 = MMFamily1(self.num_states)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta = self.delta)
        self.qmm1_0_0 = self.qmmf1_0.sample()[0]

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
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=5)

        avg_times_naive1_list = {}
        avg_times_naive2_list =  {}
        avg_times_naive3_list =  {}
        avg_times_naive4_list =  {}
        avg_times_naive5_list =  {}
        avg_times_bayes1_list =  {}
        avg_times_bayes2_list =  {}
        avg_times_bayes3_list = {}
        avg_times_bayes4_list = {}
        avg_times_bayes5_list = {}
        avg_errs_naive1_list =  {}
        avg_errs_naive2_list =  {}
        avg_errs_naive3_list =  {}
        avg_errs_naive4_list =  {}
        avg_errs_naive5_list =  {}
        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        avg_errs_bayes3_list = {}
        avg_errs_bayes4_list = {}
        avg_errs_bayes5_list = {}

        taumeta_values = []
        eta_values = []
        scale_window_values = []
        num_traj_values = []

        numruns = 1
        for i in range (0,numruns):
            # calculate performances and errors for the three parameters
            avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = self.test_taumeta_eta()
            avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = self.test_taumeta_scale_window()
            avg_times_naive3,  avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = self.test_taumeta_num_traj()
            avg_times_naive4, avg_errs_naive4, avg_times_bayes4, avg_errs_bayes4, taumeta_values, timescaledisp_values = self.test_taumeta_timescaledisp()
            avg_times_naive5, avg_errs_naive5, avg_times_bayes5, avg_errs_bayes5, taumeta_values, statconc_values = self.test_taumeta_statconc()

            avg_times_naive1_list[i] = (avg_times_naive1)
            avg_times_naive2_list[i] = (avg_times_naive2)
            avg_times_naive3_list[i] = (avg_times_naive3)
            avg_times_naive4_list[i] = (avg_times_naive4)
            avg_times_naive5_list[i] = (avg_times_naive5)

            avg_times_bayes1_list[i] = (avg_times_bayes1)
            avg_times_bayes2_list[i] = (avg_times_bayes2)
            avg_times_bayes3_list[i] = (avg_times_bayes3)
            avg_times_bayes4_list[i] = (avg_times_bayes4)
            avg_times_bayes5_list[i] = (avg_times_bayes5)

            avg_errs_naive1_list[i] = (avg_errs_naive1)
            avg_errs_naive2_list[i] = (avg_errs_naive2)
            avg_errs_naive3_list[i] = (avg_errs_naive3)
            avg_errs_naive4_list[i] = (avg_errs_naive4)
            avg_errs_naive5_list[i] = (avg_errs_naive5)

            avg_errs_bayes1_list[i] = (avg_errs_bayes1)
            avg_errs_bayes2_list[i] = (avg_errs_bayes2)
            avg_errs_bayes3_list[i] = (avg_errs_bayes3)
            avg_errs_bayes4_list[i] = (avg_errs_bayes4)
            avg_errs_bayes5_list[i] = (avg_errs_bayes5)

        avg_times_naive1 = np.mean(list(avg_times_naive1_list.values()), axis=0)
        avg_times_naive2 = np.mean(list(avg_times_naive2_list.values()), axis=0)
        avg_times_naive3 = np.mean(list(avg_times_naive3_list.values()), axis=0)
        avg_times_naive4 = np.mean(list(avg_times_naive4_list.values()), axis=0)
        avg_times_naive5 = np.mean(list(avg_times_naive5_list.values()), axis=0)

        avg_times_bayes1 = np.mean(list(avg_times_bayes1_list.values()), axis=0)
        avg_times_bayes2 = np.mean(list(avg_times_bayes2_list.values()), axis=0)
        avg_times_bayes3 = np.mean(list(avg_times_bayes3_list.values()), axis=0)
        avg_times_bayes4 = np.mean(list(avg_times_bayes4_list.values()), axis=0)
        avg_times_bayes5 = np.mean(list(avg_times_bayes5_list.values()), axis=0)


        print("NORMAL",list(avg_times_naive1_list.values()),"MEAN ARRAY",avg_times_naive1)
        print("NORMAL", list(avg_times_naive2_list.values()), "MEAN ARRAY", avg_times_naive2)
        print("NORMAL", list(avg_times_naive3_list.values()), "MEAN ARRAY", avg_times_naive3)
        print("NORMAL", list(avg_times_naive4_list.values()), "MEAN ARRAY", avg_times_naive4)
        print("NORMAL", list(avg_times_naive5_list.values()), "MEAN ARRAY", avg_times_naive5)

        print("NORMAL", list(avg_times_bayes1_list.values()), "MEAN ARRAY", avg_times_bayes1)
        print("NORMAL", list(avg_times_bayes2_list.values()), "MEAN ARRAY", avg_times_bayes2)
        print("NORMAL", list(avg_times_bayes3_list.values()), "MEAN ARRAY", avg_times_bayes3)
        print("NORMAL", list(avg_times_bayes4_list.values()), "MEAN ARRAY", avg_times_bayes4)
        print("NORMAL", list(avg_times_bayes5_list.values()), "MEAN ARRAY", avg_times_bayes5)

        # get minimum and maximum performance
        min_val = np.amin([avg_times_naive1,avg_times_naive2,avg_times_naive3, avg_times_naive4, avg_times_naive5, avg_times_bayes1,avg_times_bayes2,avg_times_bayes3, avg_times_bayes4, avg_times_bayes5])
        max_val = np.amax([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_naive4, avg_times_naive5, avg_times_bayes1,avg_times_bayes2,avg_times_bayes3,avg_times_bayes4,avg_times_bayes5])


        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_window_values, y_label="scale_window", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="num_traj", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive4, data_bayes=avg_times_bayes4,
                                        x_labels=taumeta_values, y_labels=timescaledisp_values, y_label="tdisp",
                                        minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive5, data_bayes=avg_times_bayes5,
                                        x_labels=taumeta_values, y_labels=statconc_values, y_label="conc",
                                        minimum=min_val, maximum=max_val)

        plots.save_plot_same_colorbar("Performance"+str(self.delta))


        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=5)

        avg_errs_naive1 = np.mean(list(avg_errs_naive1_list.values()), axis=0)
        avg_errs_naive2 = np.mean(list(avg_errs_naive2_list.values()), axis=0)
        avg_errs_naive3 = np.mean(list(avg_errs_naive3_list.values()), axis=0)
        avg_errs_naive4 = np.mean(list(avg_errs_naive4_list.values()), axis=0)
        avg_errs_naive5 = np.mean(list(avg_errs_naive5_list.values()), axis=0)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)
        avg_errs_bayes4 = np.mean(list(avg_errs_bayes4_list.values()), axis=0)
        avg_errs_bayes5 = np.mean(list(avg_errs_bayes5_list.values()), axis=0)

        print("NORMAL", list(avg_errs_naive1_list.values()), "MEAN ARRAY", avg_errs_naive1)
        print("NORMAL", list(avg_errs_naive2_list.values()), "MEAN ARRAY", avg_errs_naive2)
        print("NORMAL", list(avg_errs_naive3_list.values()), "MEAN ARRAY", avg_errs_naive3)
        print("NORMAL", list(avg_errs_naive4_list.values()), "MEAN ARRAY", avg_errs_naive4)
        print("NORMAL", list(avg_errs_naive5_list.values()), "MEAN ARRAY", avg_errs_naive5)

        print("NORMAL", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        print("NORMAL", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        print("NORMAL", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)
        print("NORMAL", list(avg_errs_bayes4_list.values()), "MEAN ARRAY", avg_errs_bayes4)
        print("NORMAL", list(avg_errs_bayes5_list.values()), "MEAN ARRAY", avg_errs_bayes5)

        # get minimum and maximum error
        min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_naive4,avg_errs_naive5,  avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3, avg_errs_bayes4, avg_errs_bayes5])
        max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_naive4,avg_errs_naive5, avg_errs_bayes1, avg_errs_bayes2,
                           avg_errs_bayes3,  avg_errs_bayes4, avg_errs_bayes5])

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, x_labels=taumeta_values,
                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, x_labels=taumeta_values,
                            y_labels=scale_window_values, y_label="scale_window", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3, x_labels=taumeta_values,
                            y_labels=num_traj_values, y_label="num_traj", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive4, data_bayes=avg_errs_bayes4, x_labels=taumeta_values,
                                        y_labels=timescaledisp_values, y_label="tdisp", minimum=min_val,
                                        maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive5, data_bayes=avg_errs_bayes5, x_labels=taumeta_values,
                                        y_labels=statconc_values, y_label="conc", minimum=min_val,
                                        maximum=max_val)

        plots.save_plot_same_colorbar("Error"+str(self.delta))


    def test_taumeta_timescaledisp(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        timescaledisp_values = create_value_list(Variable_Holder.min_timescaledisp, self.heatmap_size)

        for one,taumeta in enumerate(taumeta_values):
            for two,timescaledisp in enumerate(timescaledisp_values):
                np.random.seed(1)
                self.mmf1_0 = MMFamily1(self.num_states, timescaledisp=timescaledisp, statconc=Variable_Holder.mid_statconc)
                self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta=self.delta)
                self.qmm1_0_0 = self.qmmf1_0.sample()[0]

                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = Variable_Holder.mid_eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("TIMESCALEDISP", self.taumeta, self.shift, self.window_size, self.num_estimations, Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, Variable_Holder.mid_scale_window, timescaledisp, Variable_Holder.mid_statconc)

                slope_time_naive, avg_err_naive,  slope_time_bayes, avg_err_bayes  = self.test_timescaledisp_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, timescaledisp_values

    def test_timescaledisp_helper(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(
                dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            print("exception timescaledisp")
            return self.test_timescaledisp_helper()

    def test_taumeta_statconc(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        statconc_values = Variable_Holder.statconc_values#create_value_list_floats(Variable_Holder.min_statconc, self.heatmap_size)

        for one,taumeta in enumerate(taumeta_values):
            for two,statconc in enumerate(statconc_values):
                np.random.seed(1)
                self.mmf1_0 = MMFamily1(self.num_states, timescaledisp=Variable_Holder.mid_timescaledisp, statconc=statconc)
                self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta=self.delta)
                self.qmm1_0_0 = self.qmmf1_0.sample()[0]

                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = Variable_Holder.mid_eta * self.taumeta
                self.window_size = Variable_Holder.mid_scale_window * self.shift
                self.num_trajectories = Variable_Holder.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(Variable_Holder.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("STATCONC", self.taumeta, self.shift, self.window_size, self.num_estimations, Variable_Holder.len_trajectory, self.num_trajectories, Variable_Holder.mid_eta, Variable_Holder.mid_scale_window, Variable_Holder.mid_timescaledisp, statconc)

                slope_time_naive, avg_err_naive,  slope_time_bayes, avg_err_bayes  = self.test_statconc_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, statconc_values

    def test_statconc_helper(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(
                dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            print("exception statconc")
            return self.test_timescaledisp_helper()


    def test_taumeta_eta(self):

        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        eta_values = create_value_list(self.min_eta, self.heatmap_size)

        for one,taumeta in enumerate(taumeta_values):
            for two,eta in enumerate(eta_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = eta * self.taumeta
                self.window_size = self.mid_scale_window * self.shift
                self.num_trajectories = self.mid_num_trajectories

                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("ETA", self.taumeta, self.shift, self.window_size, self.num_estimations, self.len_trajectory, self.num_trajectories, eta, self.mid_scale_window, Variable_Holder.mid_timescaledisp, Variable_Holder.mid_statconc)

                slope_time_naive, avg_err_naive,  slope_time_bayes, avg_err_bayes  = self.test_eta_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes


        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, eta_values

    def test_eta_helper(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(
                dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            print("exception eta")
            return self.test_scale_window_helper()

    def test_taumeta_scale_window(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        scale_window_values = create_value_list(self.min_scale_window,self.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, scale_window in enumerate(scale_window_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)
                self.shift = (self.mid_eta) * self.taumeta
                self.window_size = scale_window * self.shift
                self.num_trajectories = self.mid_num_trajectories
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("scale_window", self.taumeta, self.shift, self.window_size, self.num_estimations, self.len_trajectory, self.num_trajectories, self.mid_eta, scale_window, Variable_Holder.mid_timescaledisp, Variable_Holder.mid_statconc)

                slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes  = self.test_scale_window_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""

        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, scale_window_values

    def test_scale_window_helper(self):
        # initialize timing and error arrays for naive and bayes
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(
            dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            print("exception scale window")
            return self.test_scale_window_helper()

    def test_taumeta_num_traj(self):
        avg_errs_bayes, avg_errs_naive, avg_times_bayes, avg_times_naive = init_time_and_error_arrays(self.heatmap_size)

        # specify values for taumeta and eta to iterate over
        taumeta_values = create_value_list(self.min_taumeta, self.heatmap_size)
        num_traj_values = create_value_list(self.min_num_trajectories, self.heatmap_size)

        for one, taumeta in enumerate(taumeta_values):
            for two, num_traj in enumerate(num_traj_values):

                # Setting taumeta and eta values and recalculate dependent variables for scaling
                self.taumeta = taumeta
                self.qmm1_0_0_scaled = self.qmm1_0_0.eval(self.taumeta)

                self.shift = (self.mid_eta) * self.taumeta
                # here we take the MINIMUM value of scale_window instead of the MIDDLE value on purpose
                self.window_size = (self.min_scale_window) * self.shift
                self.num_trajectories = num_traj
                self.len_trajectory = int(self.num_trajectories_len_trajectory_max / self.num_trajectories)
                self.num_estimations = Utility.calc_num_estimations(self.len_trajectory, self.window_size, self.shift)
                self.r = (self.window_size - self.shift) / self.window_size

                self.print_param_values("NUM_TRAJ",self.taumeta, self.shift, self.window_size, self.num_estimations, self.len_trajectory, self.num_trajectories, self.mid_eta, self.mid_scale_window, Variable_Holder.mid_timescaledisp, Variable_Holder.mid_statconc)

                slope_time_naive, avg_err_naive,  slope_time_bayes, avg_err_bayes  = self.test_num_traj_helper()

                avg_times_naive[two][one] = slope_time_naive
                avg_errs_naive[two][one] = avg_err_naive

                avg_times_bayes[two][one] = slope_time_bayes
                avg_errs_bayes[two][one] = avg_err_bayes

        """print("Naive Performance:", avg_times_naive)
        print("Bayes Performance:", avg_times_bayes)
        print("Naive Error:", avg_errs_naive)
        print("Bayes Error:", avg_errs_bayes)"""
        return avg_times_naive, avg_errs_naive, avg_times_bayes, avg_errs_bayes, taumeta_values, num_traj_values

    def test_num_traj_helper(self):
        etimenaive = np.zeros(self.num_estimations + 2, dtype=float)
        etimenaive[0] = 0
        err = np.zeros(self.num_estimations + 1, dtype=float)
        etimebayes = np.zeros(self.num_estimations + 2, dtype=float)
        errbayes = np.zeros(self.num_estimations + 1, dtype=float)
        self.data1_0_0 = []
        for i in range(0, self.num_trajectories):
            self.data1_0_0.append(self.qmm1_0_0_scaled.simulate(int(self.len_trajectory)))
        dataarray = np.asarray(self.data1_0_0)
        try:
            return self.performance_and_error_calculation(dataarray, err, errbayes, etimebayes, etimenaive)
        except:
            print("exception num trajectories")
            return self.test_num_traj_helper()

    def performance_and_error_calculation(self, dataarray, err, errbayes, etimebayes, etimenaive):
        #f = open("data.txt", "a")

        for k in range(0, self.num_estimations + 1):
            current_time = self.window_size + k * self.shift -1
            assert (current_time < np.shape(dataarray)[1])
            data0 = dataarray[:,current_time-self.window_size +1: (current_time + 1)]
            dataslice0 = []

            for i in range(0, self.num_trajectories):
                dataslice0.append(data0[i, :])
            if k==0:
                # init
                estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)

            t0 = process_time()
            C0 = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)  # count matrix for whole window
            C0 += 1e-8
            t1 = process_time()
            A0 = _tm(C0)

            etimenaive[k + 1] = t1 - t0 + etimenaive[k]
            #f.write("NAIVE "+str(etimenaive[k+1])+"\n")
            err[k] = np.linalg.norm(A0 - self.qmm1_0_0_scaled.eval(k).trans)
            if k == 0:
                ##### Bayes approach: Calculate C0 separately
                data0 = dataarray[:, 0 * self.shift: (self.window_size + 0 * self.shift)]
                dataslice0 = []
                for i in range(0, self.num_trajectories):
                    dataslice0.append(data0[i, :])

                t0 = process_time()
                C_old = estimate_via_sliding_windows(data=dataslice0, num_states=self.num_states)
                C0 += 1e-8
                etimebayes[1] = process_time() - t0
                #f.write("BAYES "+str(etimebayes[1])+"\n")
                errbayes[0] = np.linalg.norm(_tm(C_old) - self.qmm1_0_0_scaled.eval(k).trans)

            if k >= 1:
                ##### Bayes approach: Calculate C1 (and any following) usind C0 usind discounting
                data1new = dataarray[:, self.window_size + (k - 1) * self.shift - 1: (current_time + 1)]
                dataslice1new = []
                for i in range(0, self.num_trajectories):
                    dataslice1new.append(data1new[i, :])
                t0 = process_time()
                C_new = estimate_via_sliding_windows(data=dataslice1new, num_states=self.num_states)  # count matrix for just new transitions

                weight0 = self.r
                weight1 = 1.0

                C1bayes = weight0 * C_old + weight1 * C_new
                C_old = C1bayes

                t1 = process_time()
                etimebayes[k + 1] = t1 - t0 + etimebayes[k]
                #f.write("BAYES "+str(etimebayes[k+1])+"\n")
                A1bayes = _tm(C1bayes)
                errbayes[k] = np.linalg.norm(A1bayes - self.qmm1_0_0_scaled.eval(k).trans)
        #f.write("\n\n")
        #f.close()
        slope_time_naive = Utility.log_value(etimenaive[-1])
        avg_err_naive = Utility.log_value(sum(err) / len(err))
        slope_time_bayes = Utility.log_value(etimebayes[-1])
        avg_err_bayes = Utility.log_value(sum(errbayes) / len(errbayes))

        return slope_time_naive, avg_err_naive, slope_time_bayes, avg_err_bayes

    def print_param_values(self, evaluation_name, taumeta, shift, window_size, num_estimations, len_trajectory, num_trajectories, eta, scale_window, timescaledisp, statconc):
        print("Parameter Overview for " + evaluation_name+ ":")
        print("taumeta:\t", taumeta)
        print("eta:\t", eta)
        print("scale_window\t:", scale_window)
        print("shift:\t", shift)
        print("window_size:\t", window_size)
        print("num_estimations:\t", num_estimations)
        print("len_trajectory:\t", len_trajectory)
        print("num_trajectories:\t", num_trajectories)
        print("timescaledisp:\t",timescaledisp)
        print("statconc:\t",statconc)
        print("num_trajectories*len_trajectory:\t", num_trajectories*len_trajectory)
        print("NAIVE window_size * num_estimations\t", window_size * (num_estimations+1))
        print("BAYES window_size + num_estimations*shift\t", window_size + num_estimations*shift)
        print("\n")


delta_eval0 = Delta_Evaluation(delta=0)
delta_eval0.test_run_all_tests()

