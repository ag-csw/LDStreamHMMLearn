from unittest import TestCase

import numpy as np

from ldshmm.util.spectral_mm import SpectralMM


class TestSpectralMM(TestCase):
    ########################################
    # Spectral HMM Model Creation
    ########################################
    def create_spectral_MM(self, transD, transU):
        return SpectralMM(transD, transU)

    def scale_spectral_MM(self, sMM, tau):
        return sMM.scale(tau)

    def lincomb_spectral_MM(self, sMM0: SpectralMM, sMM1: SpectralMM, mu: float) -> SpectralMM:
        return sMM0.lincomb(sMM1, mu)

    def setUp(self):
        self.nstates = 3
        self.transD0 = np.array([[1.0, 0, 0],
                                 [0, 0.1, 0],
                                 [0, 0, 0.1]])

        self.transU0 = np.array([[0.8, 0.1, 0.1],
                                 [-1, 1, 0],
                                 [1, 1, -2]])

        self.tau = 10
        self.transD1 = np.array([[1.0, 0, 0],
                                 [0, 0.9, 0],
                                 [0, 0, 0.8]])

        self.transU1 = np.fliplr(self.transU0)

        self.mu = 0.3
        self.smm0 = self.create_spectral_MM(self.transD0, self.transU0)
        self.smm1 = self.create_spectral_MM(self.transD1, self.transU1)

    def test_isdiagonal(self):
        self.assertTrue(self.smm0.isdiagonal())
        self.assertTrue(self.smm1.isdiagonal())
        # FIXME: NOT WORKING: self.assertFalse(self.smm2.isdiagonal()) because the constructed matrix is
        # not a matrix of multinomials (some entries are negative, others are greater than 1)

    def test_lincomb(self):
        smm_lc = self.lincomb_spectral_MM(self.smm0, self.smm1, self.mu)
        self.assertTrue(smm_lc.isdiagonal())
        # FIXME: more tests

    def test_scale(self):
        smm0_scaled = self.scale_spectral_MM(self.smm0, self.tau)
        self.assertTrue(smm0_scaled.isdiagonal())
        # FIXME: more tests

    def test_isclose(self):
        self.assertTrue(self.smm0.isclose(self.smm0))

    def test_simulate(self):
        dtraj = self.smm0.simulate(100)
        self.assertEqual(len(dtraj), 100)
        self.assertTrue(max(dtraj) >= 0)
        self.assertTrue(max(dtraj) < self.nstates)
