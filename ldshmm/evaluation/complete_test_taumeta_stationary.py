from unittest import TestCase
from ldshmm.evaluation.evaluate_mm import MM_Evaluation


class Approach_Test(TestCase):


    def test_run_all_tests(self):
        self.evaluate = MM_Evaluation(number_of_runs=64)
        import sys
        sys.stdout = open("comparison_naive_bayes_error.txt", "w")
        # TODO write evaluation method for naive AND bayes

        self.evaluate.test_run_all_tests(evaluation_method="both",
                                         plot_heading="Comparison of Naïve and Bayes Error",
                                         plot_name="Comparison_Naive_Bayes_Error_MM",
                                         num_trajectories=64,
                                         numsims=64,
                                         print_intermediate_values=True)

        sys.stdout.close()

        sys.stdout = open("comparison_naive_bayes_performance.txt", "w")
        self.evaluate = MM_Evaluation(number_of_runs=32)
        self.evaluate.test_run_all_tests_performance(evaluation_method="both",
                                                     plot_heading="Comparison of Naïve and Bayes Performance",
                                                     plot_name="Comparison_Naive_Bayes_Performance_MM",
                                                     num_trajectories=256,
                                                     numsims=1,
                                                     print_intermediate_values=True)
        sys.stdout.close()

# change VariableHolder.max_numtrajectories = min_num_trajectories * heatmap_factor
# change VariableHolder.max_taumeta = min_taumeta * heatmap_factor