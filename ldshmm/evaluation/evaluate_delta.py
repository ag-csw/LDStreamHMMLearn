from ldshmm.util.plottings import ComplexPlot
from ldshmm.util.util_evaluation import *
from ldshmm.util.util_functionality import *

class Delta_Evaluation():

    def __init__(self, delta=0, number_of_runs=8):
        t0 = process_time()
        self.num_states = 4
        self.delta = delta

        self.timescaledisp = Variable_Holder.min_timescaledisp
        self.statconc = Variable_Holder.mid_statconc
        self.mmf1_0 = MMFamily1(self.num_states, timescaledisp=self.timescaledisp, statconc=self.statconc)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta = self.delta)
        self.qmm1_0_0 = self.qmmf1_0.sample()[0]
        self.numruns = number_of_runs
        # --> ConvexCombinationQuasiMM

    def test_run_all_tests(self):
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

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
        #avg_errs_naive4_list =  {}
        #avg_errs_naive5_list =  {}
        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        avg_errs_bayes3_list = {}
        #avg_errs_bayes4_list = {}
        #avg_errs_bayes5_list = {}

        taumeta_values = []
        eta_values = []
        scale_window_values = []
        num_traj_values = []

        evaluate = Evaluation_Holder(qmm1_0_0=self.qmm1_0_0, delta=self.delta)

        numruns = 1
        for i in range (0,numruns):
            # calculate performances and errors for the three parameters
            avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = evaluate.test_taumeta_eta()
            avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = evaluate.test_taumeta_scale_window()
            avg_times_naive3,  avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = evaluate.test_taumeta_num_traj()
            #avg_times_naive4, avg_errs_naive4, avg_times_bayes4, avg_errs_bayes4, taumeta_values, timescaledisp_values = self.test_taumeta_timescaledisp()
            #avg_times_naive5, avg_errs_naive5, avg_times_bayes5, avg_errs_bayes5, taumeta_values, statconc_values = self.test_taumeta_statconc()

            print("Err Naive", avg_errs_naive1)
            print("Err Bayes", avg_errs_bayes1)
            print("Err Naive", avg_errs_naive2)
            print("Err Bayes", avg_errs_bayes2)
            print("Err Naive", avg_errs_naive3)
            print("Err Bayes", avg_errs_bayes3)

            avg_times_naive1_list[i] = (avg_times_naive1)
            avg_times_naive2_list[i] = (avg_times_naive2)
            avg_times_naive3_list[i] = (avg_times_naive3)
            #avg_times_naive4_list[i] = (avg_times_naive4)
            #avg_times_naive5_list[i] = (avg_times_naive5)

            avg_times_bayes1_list[i] = (avg_times_bayes1)
            avg_times_bayes2_list[i] = (avg_times_bayes2)
            avg_times_bayes3_list[i] = (avg_times_bayes3)
            #avg_times_bayes4_list[i] = (avg_times_bayes4)
            #avg_times_bayes5_list[i] = (avg_times_bayes5)

            avg_errs_naive1_list[i] = (avg_errs_naive1)
            avg_errs_naive2_list[i] = (avg_errs_naive2)
            avg_errs_naive3_list[i] = (avg_errs_naive3)
            #avg_errs_naive4_list[i] = (avg_errs_naive4)
            #avg_errs_naive5_list[i] = (avg_errs_naive5)

            avg_errs_bayes1_list[i] = (avg_errs_bayes1)
            avg_errs_bayes2_list[i] = (avg_errs_bayes2)
            avg_errs_bayes3_list[i] = (avg_errs_bayes3)
            #avg_errs_bayes4_list[i] = (avg_errs_bayes4)
            #avg_errs_bayes5_list[i] = (avg_errs_bayes5)

        avg_times_naive1 = np.mean(list(avg_times_naive1_list.values()), axis=0)
        avg_times_naive2 = np.mean(list(avg_times_naive2_list.values()), axis=0)
        avg_times_naive3 = np.mean(list(avg_times_naive3_list.values()), axis=0)
        #avg_times_naive4 = np.mean(list(avg_times_naive4_list.values()), axis=0)
        #avg_times_naive5 = np.mean(list(avg_times_naive5_list.values()), axis=0)

        avg_times_bayes1 = np.mean(list(avg_times_bayes1_list.values()), axis=0)
        avg_times_bayes2 = np.mean(list(avg_times_bayes2_list.values()), axis=0)
        avg_times_bayes3 = np.mean(list(avg_times_bayes3_list.values()), axis=0)
        #avg_times_bayes4 = np.mean(list(avg_times_bayes4_list.values()), axis=0)
        #avg_times_bayes5 = np.mean(list(avg_times_bayes5_list.values()), axis=0)


        print("NORMAL ETA PERF",list(avg_times_naive1_list.values()),"MEAN ARRAY",avg_times_naive1)
        print("NORMAL SCALEWIN PERF", list(avg_times_naive2_list.values()), "MEAN ARRAY", avg_times_naive2)
        print("NORMAL NUMTRAJ PERF", list(avg_times_naive3_list.values()), "MEAN ARRAY", avg_times_naive3)
        #print("NORMAL", list(avg_times_naive4_list.values()), "MEAN ARRAY", avg_times_naive4)
        #print("NORMAL", list(avg_times_naive5_list.values()), "MEAN ARRAY", avg_times_naive5)

        print("NORMAL ETA ERR", list(avg_times_bayes1_list.values()), "MEAN ARRAY", avg_times_bayes1)
        print("NORMAL SCALEWIN ERR", list(avg_times_bayes2_list.values()), "MEAN ARRAY", avg_times_bayes2)
        print("NORMAL NUMTRAJ ERR", list(avg_times_bayes3_list.values()), "MEAN ARRAY", avg_times_bayes3)
        #print("NORMAL", list(avg_times_bayes4_list.values()), "MEAN ARRAY", avg_times_bayes4)
        #print("NORMAL", list(avg_times_bayes5_list.values()), "MEAN ARRAY", avg_times_bayes5)

        # get minimum and maximum performance
        #min_val = np.amin([avg_times_naive1,avg_times_naive2,avg_times_naive3, avg_times_naive4, avg_times_naive5, avg_times_bayes1,avg_times_bayes2,avg_times_bayes3, avg_times_bayes4, avg_times_bayes5])
        #max_val = np.amax([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_naive4, avg_times_naive5, avg_times_bayes1,avg_times_bayes2,avg_times_bayes3,avg_times_bayes4,avg_times_bayes5])
        min_val = np.amin([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2, avg_times_bayes3])
        max_val = np.amax([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2, avg_times_bayes3])
        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1, x_labels=taumeta_values, y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2, x_labels=taumeta_values, y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3, x_labels=taumeta_values, y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)
        """plots.add_to_plot_same_colorbar(data_naive=avg_times_naive4, data_bayes=avg_times_bayes4,
                                        x_labels=taumeta_values, y_labels=timescaledisp_values, y_label="tdisp",
                                        minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive5, data_bayes=avg_times_bayes5,
                                        x_labels=taumeta_values, y_labels=statconc_values, y_label="conc",
                                        minimum=min_val, maximum=max_val)
        """
        plots.save_plot_same_colorbar("Performance"+str(self.delta))


        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=3)

        avg_errs_naive1 = np.mean(list(avg_errs_naive1_list.values()), axis=0)
        avg_errs_naive2 = np.mean(list(avg_errs_naive2_list.values()), axis=0)
        avg_errs_naive3 = np.mean(list(avg_errs_naive3_list.values()), axis=0)
        #avg_errs_naive4 = np.mean(list(avg_errs_naive4_list.values()), axis=0)
        #avg_errs_naive5 = np.mean(list(avg_errs_naive5_list.values()), axis=0)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)
        #avg_errs_bayes4 = np.mean(list(avg_errs_bayes4_list.values()), axis=0)
        #avg_errs_bayes5 = np.mean(list(avg_errs_bayes5_list.values()), axis=0)

        print("NORMAL ETA ERR", list(avg_errs_naive1_list.values()), "MEAN ARRAY", avg_errs_naive1)
        print("NORMAL SCALEWIN ERR", list(avg_errs_naive2_list.values()), "MEAN ARRAY", avg_errs_naive2)
        print("NORMAL NUMTRAJ ERR", list(avg_errs_naive3_list.values()), "MEAN ARRAY", avg_errs_naive3)
        #print("NORMAL", list(avg_errs_naive4_list.values()), "MEAN ARRAY", avg_errs_naive4)
        #print("NORMAL", list(avg_errs_naive5_list.values()), "MEAN ARRAY", avg_errs_naive5)

        print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)
        #print("NORMAL", list(avg_errs_bayes4_list.values()), "MEAN ARRAY", avg_errs_bayes4)
        #print("NORMAL", list(avg_errs_bayes5_list.values()), "MEAN ARRAY", avg_errs_bayes5)

        # get minimum and maximum error
        #min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_naive4,avg_errs_naive5,  avg_errs_bayes1, avg_errs_bayes2,
        #                   avg_errs_bayes3, avg_errs_bayes4, avg_errs_bayes5])
        #max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_naive4,avg_errs_naive5, avg_errs_bayes1, avg_errs_bayes2,
        #                   avg_errs_bayes3,  avg_errs_bayes4, avg_errs_bayes5])
        min_val = np.amin(
            [avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1,
             avg_errs_bayes2,
             avg_errs_bayes3])
        max_val = np.amax(
            [avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1,
             avg_errs_bayes2, avg_errs_bayes3])

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, x_labels=taumeta_values,
                            y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, x_labels=taumeta_values,
                            y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3, x_labels=taumeta_values,
                            y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)
        """plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive4, data_bayes=avg_errs_bayes4, x_labels=taumeta_values,
                                        y_labels=timescaledisp_values, y_label="tdisp", minimum=min_val,
                                        maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive5, data_bayes=avg_errs_bayes5, x_labels=taumeta_values,
                                        y_labels=statconc_values, y_label="conc", minimum=min_val,
                                        maximum=max_val)
        """
        plots.save_plot_same_colorbar("Error"+str(self.delta))

    def evaluation_qmm(self):
        t0 = process_time()
        plots = ComplexPlot()
        plots.new_plot("Naive Performance vs. Bayes Performance", rows=3)

        avg_times_naive1_list = {}
        avg_times_naive2_list = {}
        avg_times_naive3_list = {}
        avg_times_naive4_list = {}
        avg_times_naive5_list = {}
        avg_times_bayes1_list = {}
        avg_times_bayes2_list = {}
        avg_times_bayes3_list = {}
        avg_times_bayes4_list = {}
        avg_times_bayes5_list = {}
        avg_errs_naive1_list = {}
        avg_errs_naive2_list = {}
        avg_errs_naive3_list = {}
        # avg_errs_naive4_list =  {}
        # avg_errs_naive5_list =  {}
        avg_errs_bayes1_list = {}
        avg_errs_bayes2_list = {}
        avg_errs_bayes3_list = {}
        # avg_errs_bayes4_list = {}
        # avg_errs_bayes5_list = {}

        taumeta_values = []
        eta_values = []
        scale_window_values = []
        num_traj_values = []

        numsims = 1
        evaluate = Evaluation_Holder(qmm1_0_0=self.qmm1_0_0, delta=self.delta, simulate=False)
        print("Start "+str(self.numruns)+" run(s)")
        t00 = process_time()
        data = []
        for i in range(0, self.numruns):
            print("Starting Run "+str(i))
            if i % numsims == 0:
                self.qmm1_0_0 = self.qmmf1_0.sample()[0]
            simulate_and_store_data(qmm1_0_0=self.qmm1_0_0, filename="qmm")
            simulated_data = read_simulated_data("qmm")

            # calculate performances and errors for the three parameters
            avg_times_naive1, avg_errs_naive1, avg_times_bayes1, avg_errs_bayes1, taumeta_values, eta_values = evaluate.test_taumeta_eta(simulated_data=simulated_data)
            avg_times_naive2, avg_errs_naive2, avg_times_bayes2, avg_errs_bayes2, taumeta_values, scale_window_values = evaluate.test_taumeta_scale_window(simulated_data=simulated_data)
            avg_times_naive3, avg_errs_naive3, avg_times_bayes3, avg_errs_bayes3, taumeta_values, num_traj_values = evaluate.test_taumeta_num_traj(simulated_data=simulated_data)
            # avg_times_naive4, avg_errs_naive4, avg_times_bayes4, avg_errs_bayes4, taumeta_values, timescaledisp_values = self.test_taumeta_timescaledisp()
            # avg_times_naive5, avg_errs_naive5, avg_times_bayes5, avg_errs_bayes5, taumeta_values, statconc_values = self.test_taumeta_statconc()

            print("Err Naive", avg_errs_naive1)
            print("Err Bayes", avg_errs_bayes1)
            print("Err Naive", avg_errs_naive2)
            print("Err Bayes", avg_errs_bayes2)
            print("Err Naive", avg_errs_naive3)
            print("Err Bayes", avg_errs_bayes3)

            avg_times_naive1_list[i] = (avg_times_naive1)
            avg_times_naive2_list[i] = (avg_times_naive2)
            avg_times_naive3_list[i] = (avg_times_naive3)
            # avg_times_naive4_list[i] = (avg_times_naive4)
            # avg_times_naive5_list[i] = (avg_times_naive5)

            avg_times_bayes1_list[i] = (avg_times_bayes1)
            avg_times_bayes2_list[i] = (avg_times_bayes2)
            avg_times_bayes3_list[i] = (avg_times_bayes3)
            # avg_times_bayes4_list[i] = (avg_times_bayes4)
            # avg_times_bayes5_list[i] = (avg_times_bayes5)

            avg_errs_naive1_list[i] = (avg_errs_naive1)
            avg_errs_naive2_list[i] = (avg_errs_naive2)
            avg_errs_naive3_list[i] = (avg_errs_naive3)
            # avg_errs_naive4_list[i] = (avg_errs_naive4)
            # avg_errs_naive5_list[i] = (avg_errs_naive5)

            avg_errs_bayes1_list[i] = (avg_errs_bayes1)
            avg_errs_bayes2_list[i] = (avg_errs_bayes2)
            avg_errs_bayes3_list[i] = (avg_errs_bayes3)
            # avg_errs_bayes4_list[i] = (avg_errs_bayes4)
            # avg_errs_bayes5_list[i] = (avg_errs_bayes5)

            if i == 3:

                mean_avg_errs_naiveeta = np.mean(list(avg_errs_naive1_list.values()), axis=0)
                mean_avg_errs_naivescalewin = np.mean(list(avg_errs_naive2_list.values()), axis=0)
                mean_avg_errs_naivenumtraj = np.mean(list(avg_errs_naive3_list.values()), axis=0)

                mean_avg_errs_bayeseta = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
                mean_avg_errs_bayesscalewin = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
                mean_avg_errs_bayesnumtraj = np.mean(list(avg_errs_bayes3_list.values()), axis=0)


                data.append(mean_avg_errs_naiveeta)
                data.append(mean_avg_errs_naivescalewin)
                data.append(mean_avg_errs_naivenumtraj)
                data.append(mean_avg_errs_bayeseta)
                data.append(mean_avg_errs_bayesscalewin)
                data.append(mean_avg_errs_bayesnumtraj)
                #np.savetxt("Errors0-3.txt", X=data, delimiter=",")



        print("Done with "+str(self.numruns)+" runs - "+str(process_time()-t00))
        avg_times_naive1 = np.mean(list(avg_times_naive1_list.values()), axis=0)
        avg_times_naive2 = np.mean(list(avg_times_naive2_list.values()), axis=0)
        avg_times_naive3 = np.mean(list(avg_times_naive3_list.values()), axis=0)
        # avg_times_naive4 = np.mean(list(avg_times_naive4_list.values()), axis=0)
        # avg_times_naive5 = np.mean(list(avg_times_naive5_list.values()), axis=0)

        avg_times_bayes1 = np.mean(list(avg_times_bayes1_list.values()), axis=0)
        avg_times_bayes2 = np.mean(list(avg_times_bayes2_list.values()), axis=0)
        avg_times_bayes3 = np.mean(list(avg_times_bayes3_list.values()), axis=0)
        # avg_times_bayes4 = np.mean(list(avg_times_bayes4_list.values()), axis=0)
        # avg_times_bayes5 = np.mean(list(avg_times_bayes5_list.values()), axis=0)



        print("NORMAL ETA PERF", list(avg_times_naive1_list.values()), "MEAN ARRAY", avg_times_naive1)
        print("NORMAL SCALEWIN PERF", list(avg_times_naive2_list.values()), "MEAN ARRAY", avg_times_naive2)
        print("NORMAL NUMTRAJ PERF", list(avg_times_naive3_list.values()), "MEAN ARRAY", avg_times_naive3)
        # print("NORMAL", list(avg_times_naive4_list.values()), "MEAN ARRAY", avg_times_naive4)
        # print("NORMAL", list(avg_times_naive5_list.values()), "MEAN ARRAY", avg_times_naive5)

        print("NORMAL ETA ERR", list(avg_times_bayes1_list.values()), "MEAN ARRAY", avg_times_bayes1)
        print("NORMAL SCALEWIN ERR", list(avg_times_bayes2_list.values()), "MEAN ARRAY", avg_times_bayes2)
        print("NORMAL NUMTRAJ ERR", list(avg_times_bayes3_list.values()), "MEAN ARRAY", avg_times_bayes3)
        # print("NORMAL", list(avg_times_bayes4_list.values()), "MEAN ARRAY", avg_times_bayes4)
        # print("NORMAL", list(avg_times_bayes5_list.values()), "MEAN ARRAY", avg_times_bayes5)

        # get minimum and maximum performance
        # min_val = np.amin([avg_times_naive1,avg_times_naive2,avg_times_naive3, avg_times_naive4, avg_times_naive5, avg_times_bayes1,avg_times_bayes2,avg_times_bayes3, avg_times_bayes4, avg_times_bayes5])
        # max_val = np.amax([avg_times_naive1,avg_times_naive2,avg_times_naive3,avg_times_naive4, avg_times_naive5, avg_times_bayes1,avg_times_bayes2,avg_times_bayes3,avg_times_bayes4,avg_times_bayes5])
        min_val = np.amin([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
                           avg_times_bayes3])
        max_val = np.amax([avg_times_naive1, avg_times_naive2, avg_times_naive3, avg_times_bayes1, avg_times_bayes2,
                           avg_times_bayes3])
        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive1, data_bayes=avg_times_bayes1,
                                        x_labels=taumeta_values, y_labels=eta_values, y_label="eta", minimum=min_val,
                                        maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive2, data_bayes=avg_times_bayes2,
                                        x_labels=taumeta_values, y_labels=scale_window_values, y_label="scwin",
                                        minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive3, data_bayes=avg_times_bayes3,
                                        x_labels=taumeta_values, y_labels=num_traj_values, y_label="ntraj",
                                        minimum=min_val, maximum=max_val)
        """plots.add_to_plot_same_colorbar(data_naive=avg_times_naive4, data_bayes=avg_times_bayes4,
                                        x_labels=taumeta_values, y_labels=timescaledisp_values, y_label="tdisp",
                                        minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_times_naive5, data_bayes=avg_times_bayes5,
                                        x_labels=taumeta_values, y_labels=statconc_values, y_label="conc",
                                        minimum=min_val, maximum=max_val)
        """
        plots.save_plot_same_colorbar("Performance" + str(self.delta))

        ###########################################################
        plots = ComplexPlot()
        plots.new_plot("Naive Error vs. Bayes Error", rows=3)

        avg_errs_naive1 = np.mean(list(avg_errs_naive1_list.values()), axis=0)
        avg_errs_naive2 = np.mean(list(avg_errs_naive2_list.values()), axis=0)
        avg_errs_naive3 = np.mean(list(avg_errs_naive3_list.values()), axis=0)
        # avg_errs_naive4 = np.mean(list(avg_errs_naive4_list.values()), axis=0)
        # avg_errs_naive5 = np.mean(list(avg_errs_naive5_list.values()), axis=0)

        avg_errs_bayes1 = np.mean(list(avg_errs_bayes1_list.values()), axis=0)
        avg_errs_bayes2 = np.mean(list(avg_errs_bayes2_list.values()), axis=0)
        avg_errs_bayes3 = np.mean(list(avg_errs_bayes3_list.values()), axis=0)
        # avg_errs_bayes4 = np.mean(list(avg_errs_bayes4_list.values()), axis=0)
        # avg_errs_bayes5 = np.mean(list(avg_errs_bayes5_list.values()), axis=0)

        print("NORMAL ETA ERR", list(avg_errs_naive1_list.values()), "MEAN ARRAY", avg_errs_naive1)
        print("NORMAL SCALEWIN ERR", list(avg_errs_naive2_list.values()), "MEAN ARRAY", avg_errs_naive2)
        print("NORMAL NUMTRAJ ERR", list(avg_errs_naive3_list.values()), "MEAN ARRAY", avg_errs_naive3)
        # print("NORMAL", list(avg_errs_naive4_list.values()), "MEAN ARRAY", avg_errs_naive4)
        # print("NORMAL", list(avg_errs_naive5_list.values()), "MEAN ARRAY", avg_errs_naive5)

        print("BAYES ETA ERR", list(avg_errs_bayes1_list.values()), "MEAN ARRAY", avg_errs_bayes1)
        print("BAYES SCALEWIN ERR", list(avg_errs_bayes2_list.values()), "MEAN ARRAY", avg_errs_bayes2)
        print("BAYES NUMTRAJ ERR", list(avg_errs_bayes3_list.values()), "MEAN ARRAY", avg_errs_bayes3)
        # print("NORMAL", list(avg_errs_bayes4_list.values()), "MEAN ARRAY", avg_errs_bayes4)
        # print("NORMAL", list(avg_errs_bayes5_list.values()), "MEAN ARRAY", avg_errs_bayes5)


        data2 = []

        data2.append(avg_errs_naive1)
        data2.append(avg_errs_naive2)
        data2.append(avg_errs_naive3)
        data2.append(avg_errs_bayes1)
        data2.append(avg_errs_bayes2)
        data2.append(avg_errs_bayes3)

        #np.savetxt("Errors0-7.txt",X=data2, delimiter=",")
        # get minimum and maximum error
        # min_val = np.amin([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_naive4,avg_errs_naive5,  avg_errs_bayes1, avg_errs_bayes2,
        #                   avg_errs_bayes3, avg_errs_bayes4, avg_errs_bayes5])
        # max_val = np.amax([avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_naive4,avg_errs_naive5, avg_errs_bayes1, avg_errs_bayes2,
        #                   avg_errs_bayes3,  avg_errs_bayes4, avg_errs_bayes5])
        min_val = np.amin(
            [avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1,
             avg_errs_bayes2,
             avg_errs_bayes3])
        max_val = np.amax(
            [avg_errs_naive1, avg_errs_naive2, avg_errs_naive3, avg_errs_bayes1,
             avg_errs_bayes2, avg_errs_bayes3])

        # input data into one plot
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive1, data_bayes=avg_errs_bayes1, x_labels=taumeta_values,
                                        y_labels=eta_values, y_label="eta", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive2, data_bayes=avg_errs_bayes2, x_labels=taumeta_values,
                                        y_labels=scale_window_values, y_label="scwin", minimum=min_val, maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive3, data_bayes=avg_errs_bayes3, x_labels=taumeta_values,
                                        y_labels=num_traj_values, y_label="ntraj", minimum=min_val, maximum=max_val)
        """plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive4, data_bayes=avg_errs_bayes4, x_labels=taumeta_values,
                                        y_labels=timescaledisp_values, y_label="tdisp", minimum=min_val,
                                        maximum=max_val)
        plots.add_to_plot_same_colorbar(data_naive=avg_errs_naive5, data_bayes=avg_errs_bayes5, x_labels=taumeta_values,
                                        y_labels=statconc_values, y_label="conc", minimum=min_val,
                                        maximum=max_val)
        """
        plots.save_plot_same_colorbar("Error" + str(self.delta))
        print("Done Evaluating - "+ str(process_time()-t0))

        print("Average Errors Run 1-4: ")
        print(data)
        print("Average Errors Run 1-8: ")
        print(data2)

#delta_eval0 = Delta_Evaluation(delta=0)
#delta_eval0.test_run_all_tests()

