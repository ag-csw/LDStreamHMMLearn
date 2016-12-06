from ldshmm.evaluation.evaluate_mm import MM_Evaluation

class Decile_Evaluator():

    def __init__(self):
        self.evaluate = MM_Evaluation(number_of_runs=8)

    def evaluate_deciles(self):
        self.evaluate.test_mid_values_bayes_NEW(plot_heading="Distribution of Transition Matrix Error Along Trajectory (Bayes)",
                                                plotname="Deciles_Bayes_MM",
                                                num_trajectories=32
                                                )


decile_eval = Decile_Evaluator()
decile_eval.evaluate_deciles()