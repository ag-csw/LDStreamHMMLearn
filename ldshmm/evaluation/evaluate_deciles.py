from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.variable_holder import Variable_Holder
from unittest import TestCase
import sys


class Decile_Evaluator(TestCase):

    def setUp(self):
        self.evaluate = MM_Evaluation(number_of_runs=8)

    def test_evaluate_deciles_mm(self):
        sys.stdout = open("evaluate_deciles_mm.txt", "w")
        mmf1 =MMFamily1(nstates=Variable_Holder.num_states)
        self.evaluate.test_mid_values_bayes_NEW(model=mmf1,
                                                plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_MM",
                                                num_trajectories=128,
                                                numsims=32,
                                                print_intermediate_values=True
                                                 )
        sys.stdout.close()

    def test_evaluate_deciles_qmm(self):
        sys.stdout = open("evaluate_deciles_qmm.txt", "w")
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
        sys.stdout.close()

    def test_evaluate_deciles_qmm_mu(self):
        numruns=32
        self.evaluate = MM_Evaluation(number_of_runs=numruns)
        mmf1 = MMFamily1(nstates=Variable_Holder.num_states)
        delta=0.2
        qmmf1 = QMMFamily1(mmfam=mmf1, delta=delta, edgeshift=256, edgewidth=4)
        sys.stdout = open("evaluate_deciles_qmm_mu_"+str(delta)+".txt", "w")
        print("Numruns\t",numruns)
        print("Statconc:\t", qmmf1.mmfam.statconcvec)
        print("Timescaledisp:\t", qmmf1.mmfam.timescaledisp)
        print("Edgewidth:\t", qmmf1.edgewidth)
        print("Edgeshift:\t", qmmf1.edgeshift)
        print("Gammamin:\t", qmmf1.gammamin)
        print("Gammamax:\t", qmmf1.gammamax)

        print("Delta:\t", qmmf1.delta)


        self.evaluate.test_mid_values_bayes_additional_mu_plot(model=qmmf1,
                                                plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_QMM_Mu_Delta_"+str(delta),
                                                num_trajectories=8,
                                                numsims=1,
                                                print_intermediate_values=True
                                                )
        sys.stdout.close()

# num_trajectories  = numsims * num_trajs
