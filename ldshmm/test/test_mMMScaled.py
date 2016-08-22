from unittest import TestCase

import numpy as np

from ldshmm.util.mm_class import MMMScaled
from ldshmm.util.spectral_mm import SpectralMM


class TestMMMScaled(TestCase):
    def setUp(self):
        self.transD0 = np.array([[1.0, 0, 0],
                                 [0, 0.2, 0],
                                 [0, 0, 0.1]])

        self.transU0 = np.array([[0.8, 0.1, 0.1],
                                 [-1, 1, 0],
                                 [1, 1, -2]])

        self.tau = 10
        self.smm0 = self.create_spectral_MM(self.transD0, self.transU0)
        self.mmm = self.create_scaled_MM_class(self.smm0)

    ########################################
    # Spectral MM Model Creation
    ########################################
    def create_spectral_MM(self, transD, transU):
        return SpectralMM(transD, transU)

    def scale_spectral_MM(self, sMM, tau):
        return sMM.scale(tau)

    def lincomb_spectral_MM(self, sMM0: SpectralMM, sMM1: SpectralMM, mu: float) -> SpectralMM:
        return sMM0.lincomb(sMM1, mu)

    ########################################
    # Metastable MM Class Creation
    ########################################
    def create_scaled_MM_class(self, sMM):
        return MMMScaled(sMM)

    def test_ismember(self):
        smm0_scaled2 = self.mmm.eval(self.tau)
        self.assertTrue(self.mmm.ismember(smm0_scaled2))

    def test_eval(self):
        smm0_scaled = self.scale_spectral_MM(self.smm0, self.tau)
        smm0_scaled2 = self.mmm.eval(self.tau)
        self.assertTrue(np.allclose(smm0_scaled.transition_matrix, smm0_scaled2.transition_matrix))
