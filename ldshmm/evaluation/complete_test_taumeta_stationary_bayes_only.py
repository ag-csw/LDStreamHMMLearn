from unittest import TestCase
from ldshmm.evaluation.evaluate_mm import MM_Evaluation


class Approach_Test(TestCase):
    def setUp(self):
        self.evaluate = MM_Evaluation(number_of_runs=2)

    def test_run_all_tests(self):
        self.evaluate.test_run_all_tests_bayes_only(plot_name="0")