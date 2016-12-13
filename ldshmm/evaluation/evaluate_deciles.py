from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.variable_holder import Variable_Holder
from unittest import TestCase

class Decile_Evaluator(TestCase):

    def setUp(self):
        self.evaluate = MM_Evaluation(number_of_runs=1) #1

    def test_evaluate_deciles_mm(self):
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        self.evaluate.test_mid_values_bayes_NEW(model=mmf1,
                                                plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_MM",
                                                num_trajectories=128,
                                                numsims=32,
                                                print_intermediate_values=True
                                                )

    def test_evaluate_deciles_qmm(self):
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        qmmf1 = QMMFamily1(mmfam=mmf1, delta=0)

        print("Edgewidth:\t", qmmf1.edgewidth)
        print("Edgeshift:\t", qmmf1.edgeshift)
        print("Gammamin:\t", qmmf1.gammamin)
        print("Gammamax:\t", qmmf1.gammamax)

        print("Delta:\t", qmmf1.delta)
        print("Gammadist:\t", qmmf1.gammadist)

        self.evaluate.test_mid_values_bayes_NEW(model=qmmf1,
                                               plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                               plotname="Deciles_Bayes_QMM",
                                               num_trajectories=128,
                                               numsims=32,
                                               print_intermediate_values=True
                                               )


    def test_evaluate_deciles_qmm_mu(self):
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        delta=0.1
        qmmf1 = QMMFamily1(mmfam=mmf1, delta=delta, edgeshift= 48)

        print("Edgewidth:\t", qmmf1.edgewidth)
        print("Edgeshift:\t", qmmf1.edgeshift)
        print("Gammamin:\t", qmmf1.gammamin)
        print("Gammamax:\t", qmmf1.gammamax)

        print("Delta:\t", qmmf1.delta)
        print("Gammadist:\t", qmmf1.gammadist)


        self.evaluate.test_mid_values_bayes_additional_mu_plot(model=qmmf1,
                                                plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_QMM_Mu_Delta_"+str(delta),
                                                num_trajectories=128,
                                                numsims=32,
                                                print_intermediate_values=True
                                                )

# num_trajectories  = numsims * num_trajs


# change VariableHolder.max_numtrajectories = 1
# change VariableHolder.max_taumeta = mid_taumeta