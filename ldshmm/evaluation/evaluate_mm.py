from ldshmm.util.util_functionality import *
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_evaluation_mm_bayes_only import Evaluation_Holder_MM as Evaluation_Holder_MM_Bayes_Only
from ldshmm.util.util_evaluation_mm_naive_only import Evaluation_Holder_MM as Evaluation_Holder_MM_Naive_Only
from ldshmm.util.util_evaluation_holder import Evaluation_Holder_MM as NEW_Evaluation_Holder
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.variable_configuration import Variable_Config

class MM_Evaluation():

    def __init__(self, number_of_runs=8):
        self.mmf1_0 = MMFamily1(Variable_Holder.num_states)
        self.model = self.mmf1_0.sample()[0]
        self.numruns = number_of_runs
        #simulate_and_store_data(self.mm1_0_0, "mm")

    def test_run_all_tests_bayes_only(self, plot_name=None):
        evaluate = Evaluation_Holder_MM_Bayes_Only(mm1_0_0=self.model, simulate=False)

        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        #avg_errs_bayes3_list = {}

        bayes_err_data2 = []
        bayes_err_data4 = []

        for i in range(0, self.numruns):
            print("Starting Run " + str(i))

            self.model = self.mmf1_0.sample()[0]
            self.simulated_data = simulate_and_store(model=self.model)

            num_trajs = Variable_Holder.mid_num_trajectories
            reshaped_trajs = reshape_trajs(self.simulated_data, num_trajs)
            average_complete_trajs_eta = []
            average_complete_trajs_scale_window = []
            for sub_traj in reshaped_trajs:
                # calculate performances and errors for the three parameters
                avg_errs_bayes1, taumeta_values, eta_values = evaluate.test_taumeta_eta(mm1_0_0=self.model, simulated_data=sub_traj)
                avg_errs_bayes2, taumeta_values, scale_window_values = evaluate.test_taumeta_scale_window(mm1_0_0=self.model, simulated_data=sub_traj)
                #avg_errs_bayes3, taumeta_values, num_traj_values = evaluate.test_taumeta_num_traj(mm1_0_0=self.mm1_0_0,simulated_data=simulated_data_slice)
                average_complete_trajs_eta.append(avg_errs_bayes1)
                average_complete_trajs_scale_window.append(avg_errs_bayes2)
            avg_err_eta = np.mean(average_complete_trajs_eta, axis=0)
            avg_err_scale_window = np.mean(average_complete_trajs_scale_window, axis=0)
            avg_errs_bayes1_list[i] = avg_err_eta
            avg_errs_bayes2_list[i] = avg_err_scale_window
            #avg_errs_bayes3_list[i] = (avg_errs_bayes3)


            if i == (self.numruns/4)-1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                #mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data2.append(mean_avg_errs_bayeseta)
                bayes_err_data2.append(mean_avg_errs_bayesscalewin)
                #bayes_err_data2.append(mean_avg_errs_bayesnumtraj)

            if i == (self.numruns/2)-1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                #mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data4.append(mean_avg_errs_bayeseta)
                bayes_err_data4.append(mean_avg_errs_bayesscalewin)
                #bayes_err_data4.append(mean_avg_errs_bayesnumtraj)

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Dependence of Bayes Error on Parameters", rows=2, cols=1)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        #avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

        data8=[]
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        #data8.append(avg_errs_bayes3)

        data8 = []
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        #data8.append(avg_errs_bayes3)

        print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        #print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)

        # get minimum and maximum error
        min_val = np.amin([avg_errs_bayes1, avg_errs_bayes2])#, avg_errs_bayes3])
        max_val = np.amax([avg_errs_bayes1, avg_errs_bayes2])#, avg_errs_bayes3])

        # input data into one plot
        plots.add_data_to_plot(data=avg_errs_bayes1, x_labels=taumeta_values,
                                        y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_data_to_plot(data=avg_errs_bayes2, x_labels=taumeta_values,
                                        y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        #plots.add_data_to_plot(data=avg_errs_bayes3, x_labels=taumeta_values,
        #                                y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)

        if plot_name:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_MM_delta="+plot_name)
        else:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_MM")


        print("Average Errors Run 1-"+str(int(self.numruns/4))+": ")
        print(bayes_err_data2)
        print("Average Errors Run 1-"+str(int(self.numruns/2))+": ")
        print(bayes_err_data4)
        print("Average Errors Run 1-"+str(int(self.numruns))+": ")
        print(data8)

    def test_run_all_tests_bayes_only_NEW(self, plot_name=None):

        taumeta_values = create_value_list(Variable_Holder.min_taumeta, Variable_Holder.heatmap_size)

        eta_values = create_value_list(Variable_Holder.min_eta, Variable_Holder.heatmap_size)
        variable_config_eta = Variable_Config(iter_values1=taumeta_values, iter_values2=eta_values)
        variable_config_eta.scale_window = Variable_Holder.mid_scale_window

        evaluate_eta = NEW_Evaluation_Holder(mm1_0_0=self.model, simulate=False, variable_config=variable_config_eta, evaluate_method="bayes")

        scale_window_values = create_value_list(Variable_Holder.min_scale_window, Variable_Holder.heatmap_size)
        variable_config_scale_window = Variable_Config(iter_values1=taumeta_values, iter_values2=scale_window_values)
        variable_config_scale_window.eta = Variable_Holder.mid_eta

        evaluate_scale_window = NEW_Evaluation_Holder(mm1_0_0=self.model, simulate=False, variable_config=variable_config_scale_window, evaluate_method="bayes")

        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        #avg_errs_bayes3_list = {}

        bayes_err_data2 = []
        bayes_err_data4 = []

        for i in range(0, self.numruns):
            print("Starting Run " + str(i))

            self.model = self.mmf1_0.sample()[0]
            self.simulated_data = simulate_and_store(model=self.model)

            num_trajs = Variable_Holder.mid_num_trajectories
            reshaped_trajs = reshape_trajs(self.simulated_data, num_trajs)
            average_complete_trajs_eta = []
            average_complete_trajs_scale_window = []
            for sub_traj in reshaped_trajs:
                variable_config_eta.num_trajectories = len(sub_traj)
                variable_config_scale_window.num_trajectories = len(sub_traj)
                # calculate performances and errors for the three parameters
                times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = evaluate_eta.evaluate(model=self.model, simulated_data=sub_traj)
                times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = evaluate_scale_window.evaluate(model=self.model, simulated_data=sub_traj)
                #avg_errs_bayes3, taumeta_values, num_traj_values = evaluate.test_taumeta_num_traj(mm1_0_0=self.mm1_0_0,simulated_data=simulated_data_slice)
                average_complete_trajs_eta.append(avg_errs_bayes1)
                average_complete_trajs_scale_window.append(avg_errs_bayes2)
            avg_err_eta = np.mean(average_complete_trajs_eta, axis=0)
            avg_err_scale_window = np.mean(average_complete_trajs_scale_window, axis=0)
            avg_errs_bayes1_list[i] = avg_err_eta
            avg_errs_bayes2_list[i] = avg_err_scale_window
            #avg_errs_bayes3_list[i] = (avg_errs_bayes3)


            if i == (self.numruns/4)-1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                #mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data2.append(mean_avg_errs_bayeseta)
                bayes_err_data2.append(mean_avg_errs_bayesscalewin)
                #bayes_err_data2.append(mean_avg_errs_bayesnumtraj)

            if i == (self.numruns/2)-1:
                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                #mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

                bayes_err_data4.append(mean_avg_errs_bayeseta)
                bayes_err_data4.append(mean_avg_errs_bayesscalewin)
                #bayes_err_data4.append(mean_avg_errs_bayesnumtraj)

        """avg_times_bayes1, taumeta_values, eta_values = evaluate.test_taumeta_eta_performance_only(mm1_0_0=self.mm1_0_0,simulated_data=self.simulated_data)
        avg_times_bayes2, taumeta_values, scale_window_values = evaluate.test_taumeta_scale_window_performance_only(mm1_0_0=self.mm1_0_0,simulated_data=self.simulated_data)
        avg_times_bayes3, taumeta_values, num_traj_values = evaluate.test_taumeta_num_traj_performance_only(mm1_0_0=self.mm1_0_0,simulated_data=self.simulated_data)
        """

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Dependence of Bayes Error on Parameters", rows=2, cols=1)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        #avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

        data8=[]
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        #data8.append(avg_errs_bayes3)

        data8 = []
        data8.append(avg_errs_bayes1)
        data8.append(avg_errs_bayes2)
        #data8.append(avg_errs_bayes3)

        #print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        #print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        #print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)

        # get minimum and maximum error
        min_val = np.amin([avg_errs_bayes1, avg_errs_bayes2])#, avg_errs_bayes3])
        max_val = np.amax([avg_errs_bayes1, avg_errs_bayes2])#, avg_errs_bayes3])

        # input data into one plot
        plots.add_data_to_plot(data=avg_errs_bayes1, x_labels=taumeta_values,
                                        y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_data_to_plot(data=avg_errs_bayes2, x_labels=taumeta_values,
                                        y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        #plots.add_data_to_plot(data=avg_errs_bayes3, x_labels=taumeta_values,
        #                                y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)

        if plot_name:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_MM_delta="+plot_name)
        else:
            plots.save_plot_same_colorbar("Dependence_Bayes_Error_MM")


        print("Average Errors Run 1-"+str(int(self.numruns/4))+": ")
        print(bayes_err_data2)
        print("Average Errors Run 1-"+str(int(self.numruns/2))+": ")
        print(bayes_err_data4)
        print("Average Errors Run 1-"+str(int(self.numruns))+": ")
        print(data8)

    def test_run_all_tests_timescaledisp(self):
            evaluate = Evaluation_Holder_MM(mm1_0_0=self.model, simulate=True, filename="tdisp")

            plots = ComplexPlot()
            plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

            avg_times_naive1_list = {}
            avg_times_naive2_list = {}
            avg_times_naive3_list = {}
            avg_times_bayes1_list = {}
            avg_times_bayes2_list = {}
            avg_times_bayes3_list = {}
            avg_errs_naive1_list = {}
            avg_errs_naive2_list = {}
            avg_errs_naive3_list = {}
            avg_errs_bayes1_list = {}
            avg_errs_bayes2_list = {}
            avg_errs_bayes3_list = {}

            timescaledisp_values = []
            eta_values = []
            scale_window_values = []
            num_traj_values = []

            for i in range(0, self.numruns):
                # calculate performances and errors for the three parameters
                avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, timescaledisp_values, eta_values = evaluate.test_timescaledisp_eta()
                avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, timescaledisp_values, scale_window_values = evaluate.test_timescaledisp_scale_window()

                avg_times_naive3, avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, timescaledisp_values, num_traj_values = evaluate.test_timescaledisp_num_traj()

                avg_times_naive1_list[i] = (avg_times_naive1)
                avg_times_naive2_list[i] = (avg_times_naive2)
                avg_times_naive3_list[i] = (avg_times_naive3)

                avg_times_bayes1_list[i] = (avg_times_bayes1)
                avg_times_bayes2_list[i] = (avg_times_bayes2)
                avg_times_bayes3_list[i] = (avg_times_bayes3)

                avg_errs_naive1_list[i] = (avg_errs_naive1)
                avg_errs_naive2_list[i] = (avg_errs_naive2)
                avg_errs_naive3_list[i] = (avg_errs_naive3)

                avg_errs_bayes1_list[i] = (avg_errs_bayes1)
                avg_errs_bayes2_list[i] = (avg_errs_bayes2)
                avg_errs_bayes3_list[i] = (avg_errs_bayes3)

            avg_times_naive1 = np.mean(list(avg_times_naive1_list.values()), axis=0)
            avg_times_naive2 = np.mean(list(avg_times_naive2_list.values()), axis=0)
            avg_times_naive3 = np.mean(list(avg_times_naive3_list.values()), axis=0)
            avg_times_bayes1 = np.mean(list(avg_times_bayes1_list.values()), axis=0)
            avg_times_bayes2 = np.mean(list(avg_times_bayes2_list.values()), axis=0)
            avg_times_bayes3 = np.mean(list(avg_times_bayes3_list.values()), axis=0)

            print("NORMAL ETA PERF", list(avg_times_naive1_list.values()), "MEAN ARRAY", avg_times_naive1)
            print("NORMAL SCALEWIN PERF", list(avg_times_naive2_list.values()), "MEAN ARRAY", avg_times_naive2)
            print("NORMAL NUMTRAJ PERF", list(avg_times_naive3_list.values()), "MEAN ARRAY", avg_times_naive3)
            print("BAYES ETA PERF", list(avg_times_bayes1_list.values()), "MEAN ARRAY", avg_times_bayes1)
            print("NORMAL SCALEWIN PERF", list(avg_times_bayes2_list.values()), "MEAN ARRAY", avg_times_bayes2)
            print("NORMAL NUMTRAJ PERF", list(avg_times_bayes3_list.values()), "MEAN ARRAY", avg_times_bayes3)

            # get minimum and maximum performance
            min_val = np.amin([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
                               avg_times_bayes3])
            max_val = np.amax([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
                               avg_times_bayes3])

            # input data into one plot
            plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1,
                                            x_labels=timescaledisp_values, y_labels=eta_values, y_label="eta",
                                            minimum=min_val, maximum=max_val, x_label="tdisp")
            plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2,
                                            x_labels=timescaledisp_values, y_labels=scale_window_values, y_label="scwin",
                                            minimum=min_val, maximum=max_val, x_label="tdisp")
            plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3,
                                            x_labels=timescaledisp_values, y_labels=num_traj_values, y_label="ntraj",
                                            minimum=min_val, maximum=max_val, x_label="tdisp")

            plots.save_plot_same_colorbar("Performance")

            ###########################################################
            plots = ComplexPlot()
            plots.new_plot("Naive Error vs. Bayes Error", rows=3)

            avg_errs_naive1 = np.mean(list(avg_errs_naive1_list.values()), axis=0)
            avg_errs_naive2 = np.mean(list(avg_errs_naive2_list.values()), axis=0)
            avg_errs_naive3 = np.mean(list(avg_errs_naive3_list.values()), axis=0)
            avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
            avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
            avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

            print("NORMAL ETA ERR", list(avg_errs_naive1_list.values()), "MEAN ARRAY", avg_errs_naive1)
            print("NORMAL SCALEWIN ERR", list(avg_errs_naive2_list.values()), "MEAN ARRAY", avg_errs_naive2)
            print("NORMAL NUMTRAJ ERR", list(avg_errs_naive3_list.values()), "MEAN ARRAY", avg_errs_naive3)
            print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
            print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
            print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)

            # get minimum and maximum error
            min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                               avg_errs_bayes3])
            max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                               avg_errs_bayes3])

            # input data into one plot
            plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1,
                                            x_labels=timescaledisp_values,
                                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val, x_label="tdisp")
            plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2,
                                            x_labels=timescaledisp_values,
                                            y_labels=scale_window_values, y_label="scwin", minimum=min_val,
                                            maximum=max_val, x_label="tdisp")
            plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3,
                                            x_labels=timescaledisp_values,
                                            y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val, x_label="tdisp")

            plots.save_plot_same_colorbar("Error")

    def test_run_all_tests_statconc(self):
            evaluate = Evaluation_Holder_MM(mm1_0_0=self.model, simulate=True, filename="statconc")

            plots = ComplexPlot()
            plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

            avg_times_naive1_list = {}
            avg_times_naive2_list = {}
            avg_times_naive3_list = {}
            avg_times_bayes1_list = {}
            avg_times_bayes2_list = {}
            avg_times_bayes3_list = {}
            avg_errs_naive1_list = {}
            avg_errs_naive2_list = {}
            avg_errs_naive3_list = {}
            avg_errs_bayes1_list = {}
            avg_errs_bayes2_list = {}
            avg_errs_bayes3_list = {}

            statconc_values = []
            eta_values = []
            scale_window_values = []
            num_traj_values = []

            for i in range(0, self.numruns):
                # calculate performances and errors for the three parameters
                avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, statconc_values, eta_values = evaluate.test_statconc_eta()
                avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, statconc_values, scale_window_values = evaluate.test_statconc_scale_window()

                avg_times_naive3, avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, statconc_values, num_traj_values = evaluate.test_statconc_num_traj()

                avg_times_naive1_list[i] = (avg_times_naive1)
                avg_times_naive2_list[i] = (avg_times_naive2)
                avg_times_naive3_list[i] = (avg_times_naive3)

                avg_times_bayes1_list[i] = (avg_times_bayes1)
                avg_times_bayes2_list[i] = (avg_times_bayes2)
                avg_times_bayes3_list[i] = (avg_times_bayes3)

                avg_errs_naive1_list[i] = (avg_errs_naive1)
                avg_errs_naive2_list[i] = (avg_errs_naive2)
                avg_errs_naive3_list[i] = (avg_errs_naive3)

                avg_errs_bayes1_list[i] = (avg_errs_bayes1)
                avg_errs_bayes2_list[i] = (avg_errs_bayes2)
                avg_errs_bayes3_list[i] = (avg_errs_bayes3)

            avg_times_naive1 = np.mean(list(avg_times_naive1_list.values()), axis=0)
            avg_times_naive2 = np.mean(list(avg_times_naive2_list.values()), axis=0)
            avg_times_naive3 = np.mean(list(avg_times_naive3_list.values()), axis=0)
            avg_times_bayes1 = np.mean(list(avg_times_bayes1_list.values()), axis=0)
            avg_times_bayes2 = np.mean(list(avg_times_bayes2_list.values()), axis=0)
            avg_times_bayes3 = np.mean(list(avg_times_bayes3_list.values()), axis=0)

            print("NORMAL ETA PERF", list(avg_times_naive1_list.values()), "MEAN ARRAY", avg_times_naive1)
            print("NORMAL SCALEWIN PERF", list(avg_times_naive2_list.values()), "MEAN ARRAY", avg_times_naive2)
            print("NORMAL NUMTRAJ PERF", list(avg_times_naive3_list.values()), "MEAN ARRAY", avg_times_naive3)
            print("BAYES ETA PERF", list(avg_times_bayes1_list.values()), "MEAN ARRAY", avg_times_bayes1)
            print("NORMAL SCALEWIN PERF", list(avg_times_bayes2_list.values()), "MEAN ARRAY", avg_times_bayes2)
            print("NORMAL NUMTRAJ PERF", list(avg_times_bayes3_list.values()), "MEAN ARRAY", avg_times_bayes3)

            # get minimum and maximum performance
            min_val = np.amin([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
                               avg_times_bayes3])
            max_val = np.amax([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
                               avg_times_bayes3])

            # input data into one plot
            plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1,
                                            x_labels=statconc_values, y_labels=eta_values, y_label="eta",
                                            minimum=min_val, maximum=max_val, x_label="statconc")
            plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2,
                                            x_labels=statconc_values, y_labels=scale_window_values, y_label="scwin",
                                            minimum=min_val, maximum=max_val, x_label="statconc")
            plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3,
                                            x_labels=statconc_values, y_labels=num_traj_values, y_label="ntraj",
                                            minimum=min_val, maximum=max_val, x_label="statconc")

            plots.save_plot_same_colorbar("Performance_statconc")

            ###########################################################
            plots = ComplexPlot()
            plots.new_plot("Naive Error vs. Bayes Error", rows=3)

            avg_errs_naive1 = np.mean(list(avg_errs_naive1_list.values()), axis=0)
            avg_errs_naive2 = np.mean(list(avg_errs_naive2_list.values()), axis=0)
            avg_errs_naive3 = np.mean(list(avg_errs_naive3_list.values()), axis=0)
            avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
            avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
            avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)

            print("NORMAL ETA ERR", list(avg_errs_naive1_list.values()), "MEAN ARRAY", avg_errs_naive1)
            print("NORMAL SCALEWIN ERR", list(avg_errs_naive2_list.values()), "MEAN ARRAY", avg_errs_naive2)
            print("NORMAL NUMTRAJ ERR", list(avg_errs_naive3_list.values()), "MEAN ARRAY", avg_errs_naive3)
            print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
            print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
            print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)

            # get minimum and maximum error
            min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                               avg_errs_bayes3])
            max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1, avg_errs_bayes2,
                               avg_errs_bayes3])

            # input data into one plot
            plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1,
                                            x_labels=statconc_values,
                                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val, x_label="statconc")
            plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2,
                                            x_labels=statconc_values,
                                            y_labels=scale_window_values, y_label="scwin", minimum=min_val,
                                            maximum=max_val, x_label="statconc")
            plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3,
                                            x_labels=statconc_values,
                                            y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val, x_label="statconc")

            plots.save_plot_same_colorbar("Error_statconc")

    def test_mid_values_bayes(self):
        from ldshmm.util.util_math import Utility
        evaluate = Evaluation_Holder_MM_Bayes_Only(mm1_0_0=self.model, simulate=False)
        avg_errors = []
        for i in range(0, self.numruns):
            self.model = self.mmf1_0.sample()[0]
            self.simulated_data = simulate_and_store(model=self.model, taumeta=Variable_Holder.mid_taumeta)

            errors = evaluate.test_mid_values(mm1_0_0=self.model, simulated_data=self.simulated_data)
            avg_errors.append(errors)

        avg_errors_nd = np.asarray(avg_errors)
        print(avg_errors)

        decile_values = np.zeros(shape=(10, len(avg_errors_nd[0])))
        for i in range(0, len(avg_errors_nd[0])):
            num_estimation_error_i = avg_errors_nd[:,i]
            print(num_estimation_error_i)
            deciles = Utility.calc_deciles(num_estimation_error_i)
            print(deciles)
            for j,decile in enumerate(deciles):
                decile_values[j,i] = decile

        from ldshmm.util.plottings import LinePlot
        plot = LinePlot()
        plot.new_plot(heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)", cols=1, rows=1, y_label="Deciles", x_label="Timepoints")

        for decile in decile_values:
            plot.add_line_to_plot(line_data=decile)
        legend_strings = [str(x)+" % Decile" for x in np.arange(0, 100, 10)]
        plot.add_legend(x_labels=legend_strings)
        plot.save_plot("decile_bayes")

    def test_mid_values_bayes_NEW(self, plotname=None):
        from ldshmm.util.util_math import Utility
        avg_errors = []
        for i in range(0, self.numruns):
            self.model = self.mmf1_0.sample()[0]
            self.simulated_data = simulate_and_store(model=self.model, taumeta=Variable_Holder.mid_taumeta)

            taumeta = [Variable_Holder.mid_taumeta]
            eta = [Variable_Holder.mid_eta]

            variable_config = Variable_Config(iter_values1=taumeta, iter_values2=eta)
            variable_config.scale_window=Variable_Holder.mid_scale_window
            variable_config.num_trajectories=len(self.simulated_data)
            variable_config.heatmap_size = 1

            evaluate = NEW_Evaluation_Holder(mm1_0_0=self.model, simulate=False, variable_config=variable_config, evaluate_method="bayes", log_values=False)
            _ , _ , times, errors, _ , _ = evaluate.evaluate(model=self.model, simulated_data=self.simulated_data)
            avg_errors.append(errors)

        avg_errors_nd = np.asarray(avg_errors)
        print(avg_errors)

        decile_values = np.zeros(shape=(10, len(avg_errors_nd[0])))
        for i in range(0, len(avg_errors_nd[0])):
            num_estimation_error_i = avg_errors_nd[:,i]
            print(num_estimation_error_i)
            deciles = Utility.calc_deciles(num_estimation_error_i)
            print(deciles)
            for j,decile in enumerate(deciles):
                decile_values[j,i] = decile

        from ldshmm.util.plottings import LinePlot
        plot = LinePlot()
        plot.new_plot(heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)", cols=1, rows=1, y_label="Deciles", x_label="Timepoints")

        for decile in decile_values:
            plot.add_line_to_plot(line_data=decile)
        legend_strings = [str(x)+" % Decile" for x in np.arange(0, 100, 10)]
        plot.add_legend(x_labels=legend_strings)
        if plotname:
            plot.save_plot("decile_bayes"+plotname)
        else:
            plot.save_plot("decile_bayes")

    def test_mid_values_naive(self):
        from ldshmm.util.util_math import Utility
        evaluate = Evaluation_Holder_MM_Naive_Only(mm1_0_0=self.model, simulate=False)
        avg_errors = []
        for i in range(0, self.numruns):
            self.model = self.mmf1_0.sample()[0]
            self.simulated_data = simulate_and_store(model=self.model, taumeta=Variable_Holder.mid_taumeta)

            errors = evaluate.test_mid_values(mm1_0_0=self.model, simulated_data=self.simulated_data)
            avg_errors.append(errors)

        avg_errors_nd = np.asarray(avg_errors)
        print(avg_errors)


        decile_values = np.zeros(shape=(10, len(avg_errors_nd[0])))
        for i in range(0, len(avg_errors_nd[0])):
            num_estimation_error_i = avg_errors_nd[:,i]
            print(num_estimation_error_i)
            deciles = Utility.calc_deciles(num_estimation_error_i)
            print(deciles)
            for j,decile in enumerate(deciles):
                decile_values[j,i] = decile

        from ldshmm.util.plottings import LinePlot
        plot = LinePlot()
        plot.new_plot(heading="Distribution of Transition Matrix Error Along Trajectory (Naive)", cols=1, rows=1, y_label="Deciles", x_label="Timepoints")

        for decile in decile_values:
            plot.add_line_to_plot(line_data=decile)
        legend_strings = [str(x)+" % Decile" for x in np.arange(0, 100, 10)]
        plot.add_legend(x_labels=legend_strings)
        plot.save_plot("decile_naive")