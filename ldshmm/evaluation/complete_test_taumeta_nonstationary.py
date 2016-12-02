from unittest import TestCase

from ldshmm.evaluation.evaluate_delta import Delta_Evaluation

class Approach_Test(TestCase):
    def setUp(self):
        self.evaluate = Delta_Evaluation(delta=1, number_of_runs=1)

    def test_run_all_tests(self):
        import numpy as np
        np.random.seed(1000)
        self.evaluate.test_run_all_tests(evaluation_method="both",
                                         num_trajectories=16,
                                         plot_heading="Comparison of Naïve and Bayes Error",
                                         plot_name="Comparison_Naïve_Bayes_Error_QMM"
                                        )