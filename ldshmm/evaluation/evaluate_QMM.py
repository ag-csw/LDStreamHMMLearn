from ldshmm.evaluation.evaluate_delta import Delta_Evaluation
from time import process_time

class Evaluation_QMM():

    def run_qmm(self):
        t1 = process_time()
        qmm_eval = Delta_Evaluation(delta=0)
        qmm_eval.evaluation_qmm()
        print(process_time() - t1)

compare = Evaluation_QMM()
compare.run_qmm()