from ldshmm.evaluation.evaluate_delta import Delta_Evaluation
from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from time import process_time


class Comparison_of_QMM_MM():

    def run_qmm(self):
        t1 = process_time()
        qmm_eval = Delta_Evaluation(delta=0)
        qmm_eval.test_run_all_tests()

        mm_eval = MM_Evaluation()
        mm_eval.test_run_all_tests()
        print(process_time() - t1)

compare = Comparison_of_QMM_MM()
compare.run_qmm()