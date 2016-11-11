from ldshmm.util.util_functionality import *
from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_evaluation_mm import Evaluation_Holder_MM
from ldshmm.util.mm_family import MMFamily1

class MM_Evaluation():

    def __init__(self, number_of_runs=8):
        self.mmf1_0 = MMFamily1(Variable_Holder.num_states)
        self.mm1_0_0 = self.mmf1_0.sample()[0]
        self.numruns = number_of_runs
        #simulate_and_store_data(self.mm1_0_0, "mm")

    def test_run_all_tests(self):
        evaluate = Evaluation_Holder_MM(mm1_0_0=self.mm1_0_0, simulate=False)

        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

        avg_times_naive1_list = {}
        avg_times_naive2_list =  {}
        avg_times_naive3_list =  {}
        avg_times_bayes1_list =  {}
        avg_times_bayes2_list =  {}
        avg_times_bayes3_list = {}
        avg_errs_naive1_list =  {}
        avg_errs_naive2_list =  {}
        avg_errs_naive3_list =  {}
        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        avg_errs_bayes3_list = {}

        taumeta_values = []
        eta_values = []
        scale_window_values = []
        num_traj_values = []

        for i in range (0,self.numruns):
            # calculate performances and errors for the three parameters
            avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = evaluate.test_taumeta_eta()
            avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = evaluate.test_taumeta_scale_window()

            avg_times_naive3,  avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = evaluate.test_taumeta_num_traj()

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

        print("NORMAL ETA PERF",list(avg_times_naive1_list.values()),"MEAN ARRAY",avg_times_naive1)
        print("NORMAL SCALEWIN PERF", list(avg_times_naive2_list.values()), "MEAN ARRAY", avg_times_naive2)
        print("NORMAL NUMTRAJ PERF", list(avg_times_naive3_list.values()), "MEAN ARRAY", avg_times_naive3)
        print("BAYES ETA PERF", list(avg_times_bayes1_list.values()), "MEAN ARRAY", avg_times_bayes1)
        print("NORMAL SCALEWIN PERF", list(avg_times_bayes2_list.values()), "MEAN ARRAY", avg_times_bayes2)
        print("NORMAL NUMTRAJ PERF", list(avg_times_bayes3_list.values()), "MEAN ARRAY", avg_times_bayes3)

        # get minimum and maximum performance
        min_val = np.amin([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3])
        max_val = np.amax([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_bayes1,avg_times_bayes2,avg_times_bayes3])


        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)

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
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, x_labels=taumeta_values,
                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, x_labels=taumeta_values,
                            y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3, x_labels=taumeta_values,
                            y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)


        plots.save_plot_same_colorbar("Error")


    def test_run_all_tests_timescaledisp(self):
            evaluate = Evaluation_Holder_MM(mm1_0_0=self.mm1_0_0, simulate=False)

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