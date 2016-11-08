from unittest import TestCase
from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.util_functionality import *
import numpy as np


class Test_QMM_MM(TestCase):

    def setUp(self):
        self.num_states = 4
        self.create_MM()
        self.create_qMM()

    def create_qMM(self):
        self.delta = 0

        self.timescaledisp = 2
        self.statconc = 1
        self.tmp_mmf1_0 = MMFamily1(self.num_states, timescaledisp=self.timescaledisp, statconc=self.statconc)
        self.qmmf1_0 = QMMFamily1(self.tmp_mmf1_0, delta=self.delta)
        np.random.seed(1101)
        self.qmm1_0_0 = self.qmmf1_0.sample()[0]

    def create_MM(self):
        self.mmf1_0 = MMFamily1(self.num_states)
        np.random.seed(1101)
        self.mmf1_0_0 = self.mmf1_0.sample()[0]

    def test_check_same_trajectories(self):
        np.random.seed(1101)
        #simulate_and_store_data(self.qmm1_0_0, "qmm")
        simul_qmm = read_simulated_data("qmm")

        np.random.seed(1101)
        #simulate_and_store_data(self.mmf1_0_0, "mm")
        simul_mm = read_simulated_data("mm")

        for key, value in simul_mm.items():
            print("Taumeta = "+str(key))
            print(simul_qmm[key], value)
            np.testing.assert_array_equal(simul_qmm[key], value)


    def test_check_same_spectrals(self):
        spectral_qmm0 = self.qmm1_0_0.mMM0.sMM
        spectral_mm = self.mmf1_0_0.sMM

        print(spectral_qmm0.trans, spectral_mm.trans)
        np.testing.assert_array_equal(spectral_qmm0.trans, spectral_mm.trans)