from unittest import TestCase

import numpy as np
from ldshmm.util.variable_holder import Variable_Holder
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1


class Test_QMMScaled(TestCase):
    def setUp(self):
        self.num_states = 4
        self.delta = 0

        np.random.seed(1011)
        self.timescaledisp = Variable_Holder.min_timescaledisp
        self.statconc = Variable_Holder.mid_statconc
        self.mmf1_0 = MMFamily1(self.num_states)
        self.mm1_0_0 = self.mmf1_0.sample()[0]
        np.random.seed(1011)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta = self.delta)
        self.convexCombinationQuasiMM = self.qmmf1_0.sample()[0]

    def test_qmm_samples(self):
        np.random.seed(1011)
        onesample = self.qmmf1_0.sample()[0]
        np.random.seed(1011)
        twosamples = self.qmmf1_0.sample()[0]
        onesample_scaled = onesample.eval(2).sMM0
        twosamples_scaled = twosamples.eval(2).sMM0
        np.testing.assert_array_equal(onesample_scaled.trans, twosamples_scaled.trans)

    def test_mm_samples(self):
        np.random.seed(1011)
        onesample = self.mmf1_0.sample()[0]
        np.random.seed(1011)
        twosamples = self.mmf1_0.sample(2)[0]
        np.testing.assert_array_equal(onesample.sMM.trans, twosamples.sMM.trans)

    def test_ccnsmm_smms(self):
        ConvexCombinationNSMM = self.convexCombinationQuasiMM.eval(2)

        spectral1 = ConvexCombinationNSMM.sMM0
        print(type(spectral1))
        spectral_mm = self.mm1_0_0.sMM
        print(type(spectral_mm))

        np.testing.assert_array_equal(spectral1.trans, spectral_mm.trans)

    def test_samples_mm_qmm(self):
        #print("before setting the seed:", np.random.get_state())
        np.random.seed(1011)
        #print("after setting the seed:", np.random.get_state())
        self.timescaledisp = Variable_Holder.min_timescaledisp
        self.statconc = Variable_Holder.mid_statconc
        self.mmf1_0 = MMFamily1(self.num_states)
        self.mm1_0_0 = self.mmf1_0.sample()[0]

        np.random.seed(1011)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0, delta=0)
        self.convexCombinationQuasiMM = self.qmmf1_0.sample()[0]

        mm_scaled = self.mm1_0_0.eval(2)
        qmm_scaled = self.convexCombinationQuasiMM.eval(2)
        np.testing.assert_array_equal(mm_scaled.trans, qmm_scaled.eval(0).trans)


    def test_transition_matrices(self):
        taumeta_values = [np.random.randint(1,100) for i in range (0,5)]
        tauquasi_values = [np.random.randint(1,100) for i in range (0,5)]
        for taumeta in taumeta_values:
            for tauquasi in tauquasi_values:
                self.qmm1_0_0_scaled = self.convexCombinationQuasiMM.eval(taumeta, tauquasi)
                ground_truth1 = np.random.randint(1, 100)
                ground_truth2 = np.random.randint(1, 100)
                print(ground_truth1, ground_truth2)
                trans1 = self.qmm1_0_0_scaled.eval(ground_truth_time=ground_truth1).trans
                trans2 = self.qmm1_0_0_scaled.eval(ground_truth_time=ground_truth2).trans

                np.testing.assert_array_equal(trans1, trans2)
