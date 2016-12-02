from unittest import TestCase
from ldshmm.evaluation.evaluate_mm import MM_Evaluation


class Approach_Test(TestCase):
    def setUp(self):
        self.evaluate = MM_Evaluation(number_of_runs=32)

    def test_run_all_tests(self):
        # TODO write evaluation method for naive AND bayes
        """
        self.evaluate.test_run_all_tests(evaluation_method="both",
                                         plot_heading="Comparison of Naïve and Bayes Error",
                                         plot_name="Comparison_Naive_Bayes_Error_MM",
                                         num_trajectories=64)
        """

        self.evaluate.test_run_all_tests_performance(evaluation_method="both",
                                                     plot_heading="Comparison of Naïve and Bayes Performance",
                                                     plot_name="Comparison_Naive_Bayes_Performance_MM",
                                                     num_trajectories=64)
