from unittest import TestCase

from ldshmm.evaluation.evaluate_delta import Delta_Evaluation

class Approach_Test(TestCase):
    def setUp(self):
        self.evaluate = Delta_Evaluation()

    def test_run_all_tests(self):
        self.evaluate.test_run_all_tests(evaluation_method="both")