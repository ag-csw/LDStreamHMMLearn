from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_evaluation_bayes_only import Evaluation_Holder as Evaluation_Holder_Bayes_Only
from ldshmm.util.util_functionality import *
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.variable_configuration import Variable_Config
from ldshmm.util.util_evaluation_holder import Evaluation_Holder as NEW_Evaluation_Holder

class Delta_Evaluation():

    def __init__(self, delta=0, number_of_runs=8):
        self.num_states = 4
        self.delta = delta

        self.timescaledisp = Variable_Holder.min_timescaledisp
        self.statconc = Variable_Holder.mid_statconc
        self.mmf1_0 = MMFamily1(self.num_states, timescaledisp=self.timescaledisp, statconc=self.statconc)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta = self.delta)
        self.model = self.qmmf1_0.sample()[0]
        self.numruns = number_of_runs
        # --> ConvexCombinationQuasiMM


    def test_run_all_tests_bayes_only_NEW(self, plot_name=None, num_trajectories = None):

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)

        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)
        variable_config_eta = Variable_Config(iter_values1=taumeta_values, iter_values2=eta_values)
        variable_config_eta.scale_window = Variable_Holder.mid_scale_window

        evaluate_eta = NEW_Evaluation_Holder(model=self.model, simulate=False, variable_config=variable_config_eta,
                                             evaluate_method="bayes")

        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)
        variable_config_scale_window = Variable_Config(iter_values1=taumeta_values, iter_values2=scale_window_values)
        variable_config_scale_window.eta = Variable_Holder.mid_eta

        evaluate_scale_window = NEW_Evaluation_Holder(model=self.model, simulate=False,
                                                      variable_config=variable_config_scale_window,
                                                      evaluate_method="bayes")

        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        # avg_errs_bayes3_list = {}

        bayes_err_data2 = []
        bayes_err_data4 = []

        # numsims = 1
        for i in range(0, self.numruns):
            print("Starting Run " + str(i))

            self.model = self.qmmf1_0.sample()[0]
            if num_trajectories is not None:
                self.simulated_data = simulate_and_store(model=self.model, num_trajs_simulated=num_trajectories)
            else:
                self.simulated_data = simulate_and_store(model=self.model)

            num_trajs = Variable_Holder.mid_num_trajectories
            reshaped_trajs = reshape_trajs(self.simulated_data, num_trajs)
            average_complete_trajs_eta = []
            average_complete_trajs_scale_window = []
            for sub_traj in reshaped_trajs:
                variable_config_eta.num_trajectories = len(sub_traj)
                variable_config_scale_window.num_trajectories = len(sub_traj)
                # calculate performances and errors for the three parameters
                _, _, times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = evaluate_eta.evaluate(model=self.model, simulated_data=sub_traj)
                _, _, times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = evaluate_scale_window.evaluate(self.model, simulated_data=sub_traj)
                #times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = evaluate_num_traj.evaluate(self.model, simulated_data=sub_traj)
                average_complete_trajs_eta.append(avg_errs_bayes1)
                average_complete_trajs_scale_window.append(avg_errs_bayes2)
            avg_err_eta = np.mean(average_complete_trajs_eta, axis=0)
            avg_err_scale_window = np.mean(average_complete_trajs_scale_window, axis=0)
            avg_errs_bayes1_list[i] = avg_err_eta
            avg_errs_bayes2_list[i] = avg_err_scale_window
            # avg_errs_bayes3_list[i] = (avg_errs_bayes3)


            if i == (self.numruns / 4) - 1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                # mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data2.append(mean_avg_errs_bayeseta)
                bayes_err_data2.append(mean_avg_errs_bayesscalewin)
                # bayes_err_data2.append(mean_avg_errs_bayesnumtraj)

            if i == (self.numruns / 2) - 1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                # mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data4.append(mean_avg_errs_bayeseta)
                bayes_err_data4.append(mean_avg_errs_bayesscalewin)
                # bayes_err_data4.append(mean_avg_errs_bayesnumtraj)

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Dependence of Bayes Error on Parameters", rows=2, cols=1)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        # avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

        data8 = []
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        # data8.append(avg_errs_bayes3)

        data8 = []
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        # data8.append(avg_errs_bayes3)

        print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        # print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)

        # get minimum and maximum error
        min_val = np.amin(
            [avg_errs_bayes1, avg_errs_bayes2])  # , avg_errs_bayes3])
        max_val = np.amax(
            [avg_errs_bayes1, avg_errs_bayes2])  # , avg_errs_bayes3])

        # input data into one plot
        plots.add_data_to_plot(data=avg_errs_bayes1,
                               x_labels=taumeta_values,
                               y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_data_to_plot(data=avg_errs_bayes2,
                               x_labels=taumeta_values,
                               y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        # plots.add_data_to_plot(data=avg_errs_bayes3,
        #                                x_labels=taumeta_values,
        #                                y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)

        if plot_name:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_QMM_delta_NEW=" + str(plot_name))
        else:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_QMM_delta")

        print("Average Errors Run 1-" + str(int(self.numruns / 4)) + ": ")
        print(bayes_err_data2)
        print("Average Errors Run 1-" + str(int(self.numruns / 2)) + ": ")
        print(bayes_err_data4)
        print("Average Errors Run 1-" + str(int(self.numruns)) + ": ")
        print(data8)

    def test_run_all_tests(self, evaluation_method, plot_heading, plot_name=None, num_trajectories=None):

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)

        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)
        variable_config_eta = Variable_Config(iter_values1=taumeta_values, iter_values2=eta_values)
        variable_config_eta.scale_window = Variable_Holder.mid_scale_window

        evaluate_eta = NEW_Evaluation_Holder(model=self.model, simulate=False, variable_config=variable_config_eta,
                                             evaluate_method=evaluation_method)

        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)
        variable_config_scale_window = Variable_Config(iter_values1=taumeta_values, iter_values2=scale_window_values)
        variable_config_scale_window.eta = Variable_Holder.mid_eta

        evaluate_scale_window = NEW_Evaluation_Holder(model=self.model, simulate=False,
                                                      variable_config=variable_config_scale_window,
                                                      evaluate_method=evaluation_method)

        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        avg_errs_naive1_list = {}
        avg_errs_naive2_list = {}
        # avg_errs_bayes3_list = {}

        bayes_err_data2 = []
        bayes_err_data4 = []
        naive_err_data2 = []
        naive_err_data4 = []

        # numsims = 1
        for i in range(0, self.numruns):
            print("Starting Run " + str(i))

            self.model = self.qmmf1_0.sample()[0]
            if num_trajectories is not None:
                self.simulated_data = simulate_and_store(model=self.model, num_trajs_simulated=num_trajectories)
            else:
                self.simulated_data = simulate_and_store(model=self.model)

            num_trajs = num_trajectories # Variable_Holder.mid_num_trajectories
            reshaped_trajs = reshape_trajs(self.simulated_data, num_trajs)

            average_err_complete_trajs_eta = []
            average_err_complete_trajs_eta_naive = []
            average_err_complete_trajs_scale_window = []
            average_err_complete_trajs_scale_window_naive = []

            for sub_traj in reshaped_trajs:
                variable_config_eta.num_trajectories = len(sub_traj)
                variable_config_scale_window.num_trajectories = len(sub_traj)
                # calculate performances and errors for the three parameters
                times_naive1, avg_errs_naive_1, times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = evaluate_eta.evaluate(
                    model=self.model, simulated_data=sub_traj)
                times_naive2, avg_errs_naive_2, times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = evaluate_scale_window.evaluate(
                    self.model, simulated_data=sub_traj)
                # times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = evaluate_num_traj.evaluate(self.model, simulated_data=sub_traj)
                average_err_complete_trajs_eta.append(avg_errs_bayes1)
                average_err_complete_trajs_scale_window.append(avg_errs_bayes2)

                average_err_complete_trajs_eta_naive.append(avg_errs_naive_1)
                average_err_complete_trajs_scale_window_naive.append(avg_errs_naive_2)

            avg_err_eta = np.mean(average_err_complete_trajs_eta, axis=0)
            avg_err_scale_window = np.mean(average_err_complete_trajs_scale_window, axis=0)

            avg_err_eta_naive = np.mean(average_err_complete_trajs_eta_naive, axis=0)
            avg_err_scale_window_naive = np.mean(average_err_complete_trajs_scale_window_naive, axis=0)

            avg_errs_bayes1_list[i] = avg_err_eta
            avg_errs_bayes2_list[i] = avg_err_scale_window
            # avg_errs_bayes3_list[i] = (avg_errs_bayes3)
            avg_errs_naive1_list[i] = avg_err_eta_naive
            avg_errs_naive2_list[i] = avg_err_scale_window_naive

            if i == (self.numruns / 4) - 1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                # mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data2.append(mean_avg_errs_bayeseta)
                bayes_err_data2.append(mean_avg_errs_bayesscalewin)
                # bayes_err_data2.append(mean_avg_errs_bayesnumtraj)

                mean_avg_errs_naiveeta = np.mean(list(avg_errs_naive1_list.values()), axis=0)
                mean_avg_errs_naivescalewin = np.mean(list(avg_errs_naive2_list.values()), axis=0)

                naive_err_data2.append(mean_avg_errs_naiveeta)
                naive_err_data2.append(mean_avg_errs_naivescalewin)

            if i == (self.numruns / 2) - 1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                # mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data4.append(mean_avg_errs_bayeseta)
                bayes_err_data4.append(mean_avg_errs_bayesscalewin)
                # bayes_err_data4.append(mean_avg_errs_bayesnumtraj)

                mean_avg_errs_naiveeta = np.mean(list(avg_errs_naive1_list.values()), axis=0)
                mean_avg_errs_naivescalewin = np.mean(list(avg_errs_naive2_list.values()), axis=0)

                naive_err_data4.append(mean_avg_errs_naiveeta)
                naive_err_data4.append(mean_avg_errs_naivescalewin)

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot(plot_heading, rows=2, cols=2)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        # avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

        avg_errs_naive1 = np.mean(list(avg_errs_naive1_list.values()), axis=0)
        avg_errs_naive2 = np.mean(list(avg_errs_naive2_list.values()), axis=0)

        data8 = []
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        # data8.append(avg_errs_bayes3)

        data8_naive = []
        data8_naive.append(avg_errs_naive1)
        data8_naive.append(avg_errs_naive2)
        # data8.append(avg_errs_bayes3)

        print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        # print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)

        # get minimum and maximum error
        min_val = np.amin(
            [avg_errs_bayes1, avg_errs_bayes2, avg_errs_naive1, avg_errs_naive2])  # , avg_errs_bayes3])
        max_val = np.amax(
            [avg_errs_bayes1, avg_errs_bayes2, avg_errs_naive1, avg_errs_naive2])  # , avg_errs_bayes3])

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, maximum=max_val, minimum=min_val, x_labels=taumeta_values, y_labels=eta_values, y_label="eta")
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, maximum=max_val, minimum=min_val, x_labels=taumeta_values, y_labels=scale_window_values, y_label="scale_window")

        if plot_name:
            plots.save_plot_same_colorbar(str(plot_name))
        else:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_QMM_delta")

        print("Average Errors Run 1-" + str(int(self.numruns / 4)) + ": ")
        print(bayes_err_data2)
        print("Average Errors Run 1-" + str(int(self.numruns / 2)) + ": ")
        print(bayes_err_data4)
        print("Average Errors Run 1-" + str(int(self.numruns)) + ": ")
        print(data8)

