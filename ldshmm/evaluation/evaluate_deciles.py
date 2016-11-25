from ldshmm.evaluation.evaluate_mm import MM_Evaluation

class Decile_Evaluator():

    def __init__(self):
        self.evaluate = MM_Evaluation()

    def evaluate_deciles(self):
        self.evaluate.test_mid_values_bayes()
        self.evaluate.test_mid_values_naive()

decile_eval = Decile_Evaluator()
decile_eval.evaluate_deciles()