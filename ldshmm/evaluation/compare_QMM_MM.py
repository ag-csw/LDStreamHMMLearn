from ldshmm.evaluation.evaluate_delta import Delta_Evaluation
from ldshmm.evaluation.evaluate_mm import MM_Evaluation

class Comparison_of_QMM_MM():

    def run_qmm(self):
        qmm_eval = Delta_Evaluation(delta=0)
        qmm_eval.test_run_all_tests()
        spectral_mm = qmm_eval.qmm1_0_0.mMM0

        mm_eval = MM_Evaluation(mm1_0_0=spectral_mm)
        mm_eval.test_run_all_tests()

compare = Comparison_of_QMM_MM()
compare.run_qmm()