from ldshmm.evaluation.evaluate_mm import MM_Evaluation

class Decile_Evaluator():

    def __init__(self):
        self.evaluate = MM_Evaluation(number_of_runs=64)

    def evaluate_deciles(self):
        self.evaluate.test_mid_values_bayes()
        self.evaluate.test_mid_values_bayes_NEW("_NEW")

        #self.evaluate.test_mid_values_naive()

decile_eval = Decile_Evaluator()
decile_eval.evaluate_deciles()