from unittest import TestCase
from ldshmm.evaluation.evaluate_mm import MM_Evaluation
from time import process_time


class Approach_Test(TestCase):
    def setUp(self):
        self.evaluate = MM_Evaluation(number_of_runs=16)

    def test_run_all_tests(self):
        t0 = process_time()
        self.evaluate.test_run_all_tests_bayes_only(plot_name="0_16")
        print(str(process_time()-t0))

        self.evaluate.test_run_all_tests_bayes_only_NEW(plot_name="0_16_New")