from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from time import process_time
import numpy as np
from unittest import TestCase
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.variable_holder import Variable_Holder
import sys

class Comparison_of_QMM_MM(TestCase):

    def test_run_qmm_mm(self):
        num_states = Variable_Holder.num_states
        mmf = MMFamily1(nstates=num_states)
        qmmf = QMMFamily1(mmfam=mmf, delta=0)
        t1 = process_time()
        numruns=1
        num_trajectories = 1024

        sys.stdout = open("qmm_error_delta0.txt", "w")
        np.random.seed(1011)
        qmm_eval = MM_Evaluation(number_of_runs=numruns)
        qmm_eval.test_run_all_tests(model=qmmf,
                                    evaluation_method="bayes",
                                    plot_heading="QMM Error with Delta=0",
                                    plot_name="QMM_Error",
                                    num_trajectories=num_trajectories,
                                    numsims=256,
                                    print_intermediate_values=True)
        sys.stdout.close()

        sys.stdout = open("mm_error.txt", "w")
        np.random.seed(1011)
        mm_eval = MM_Evaluation(number_of_runs=numruns)
        mm_eval.test_run_all_tests(model=mmf,
                                    evaluation_method="bayes",
                                    plot_heading="MM Error",
                                    plot_name="MM_Error",
                                    num_trajectories=num_trajectories,
                                    numsims=256,
                                    print_intermediate_values=True)

        spectral_qmm0 = qmm_eval.model.mMM0.sMM
        spectral_mm = mm_eval.model.sMM
        np.testing.assert_array_equal(spectral_qmm0.trans, spectral_mm.trans)

        print(process_time() - t1)
        sys.stdout.close()
