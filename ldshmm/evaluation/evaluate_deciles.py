from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.variable_holder import Variable_Holder


class Decile_Evaluator():

    def __init__(self):
        self.evaluate = MM_Evaluation(number_of_runs=8) #1

    def evaluate_deciles_mm(self):
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        self.model = mmf1.sample()[0]
        self.evaluate.test_mid_values_bayes_NEW(model=self.model,
                                                plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_MM",
                                                num_trajectories=128,
                                                numsims=32,
                                                print_intermediate_values=True
                                                )

    def evaluate_deciles_qmm(self):
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        qmmf1 = QMMFamily1(mmfam=mmf1, delta=0)

        print("Edgewidth:\t", qmmf1.edgewidth)
        print("Edgeshift:\t", qmmf1.edgeshift)
        print("Gammamin:\t", qmmf1.gammamin)
        print("Gammamax:\t", qmmf1.gammamax)

        print("Delta:\t", qmmf1.delta)
        print("Gammadist:\t", qmmf1.gammadist)

        self.model = qmmf1.sample()[0]
        self.evaluate.test_mid_values_bayes_NEW(model=self.model,
                                               plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                               plotname="Deciles_Bayes_QMM",
                                               num_trajectories=128,
                                               numsims=32,
                                               print_intermediate_values=True
                                               )


    def evaluate_deciles_qmm_mu(self):
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        qmmf1 = QMMFamily1(mmfam=mmf1, delta=0, edgeshift=100)

        print("Edgewidth:\t", qmmf1.edgewidth)
        print("Edgeshift:\t", qmmf1.edgeshift)
        print("Gammamin:\t", qmmf1.gammamin)
        print("Gammamax:\t", qmmf1.gammamax)

        print("Delta:\t", qmmf1.delta)
        print("Gammadist:\t", qmmf1.gammadist)

        self.model = qmmf1.sample()[0]
        self.evaluate.test_mid_values_bayes_additional_mu_plot(model=self.model,
                                                plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_QMM_Mu",
                                                num_trajectories=8, #1024
                                                numsims=1, #256
                                                print_intermediate_values=True
                                                )

# num_trajectories  = numsims * num_trajs
import sys
decile_eval = Decile_Evaluator()
#sys.stdout = open("evaluate_deciles_mm.txt", "w")
#decile_eval.evaluate_deciles_mm()
#sys.stdout.close()
"""
sys.stdout = open("evaluate_deciles_qmm.txt", "w")
decile_eval.evaluate_deciles_qmm()
sys.stdout.close()
"""
sys.stdout = open("evaluate_deciles_qmm_mu.txt", "w")
decile_eval.evaluate_deciles_qmm_mu()
sys.stdout.close()


# change VariableHolder.max_numtrajectories = 1
# change VariableHolder.max_taumeta = mid_taumeta