from unittest import TestCase
from ldshmm.util.mm_family import MMFamily1
from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from ldshmm.util.util_functionality import *
from ldshmm.util.variable_configuration import Variable_Config
from ldshmm.util.util_evaluation_holder import Evaluation_Holder as NEW_Evaluation_Holder

class Test_Error_Decreases(TestCase):

    def setUp(self):
        nstates=4
        self.mmf1 = MMFamily1(nstates)
        self.numruns=1

    def test_error_increase(self):
        num_trajs = [8]#[1,2,4,8]
        numsims=1

        for num_trajectories in num_trajs:

            for i in range(0, self.numruns):
                self.model = self.mmf1.sample()[0]
                self.simulated_data = simulate_and_store(model=self.model, taumeta=Variable_Holder.mid_taumeta,
                                                         num_trajs_simulated=num_trajectories)
                num_trajs = int(num_trajectories / numsims)

                reshaped_trajs = reshape_trajs(self.simulated_data, num_trajs)
                for k, sub_traj in enumerate(reshaped_trajs):
                    taumeta = [Variable_Holder.mid_taumeta]
                    eta = [Variable_Holder.mid_eta]

                    variable_config = Variable_Config(iter_values1=taumeta, iter_values2=eta)
                    variable_config.num_trajectories = len(sub_traj)
                    variable_config.scale_window = Variable_Holder.mid_scale_window
                    variable_config.heatmap_size = 1

                    evaluate = NEW_Evaluation_Holder(model=self.model, simulate=False, variable_config=variable_config,
                                                     evaluate_method="bayes", log_values=False, avg_values=False,
                                                     heatmap=False)
                    _, _, times, errors, _, _ = evaluate.evaluate(simulated_data=sub_traj,
                                                                  print_intermediate_values=True)
                    print("Errors Numtrajectories "+str(num_trajectories), errors)
                    # print("AVG_ERRORS",np.shape(avg_errors))


