from unittest import TestCase

import numpy as np

from ldshmm.util.spectral_hmm import SpectralHMM


class TestSpectralHMM(TestCase):
    ########################################
    # Spectral HMM Model Creation
    ########################################
    def create_spectral_HMM(self, transD, transU, pobs):
        return SpectralHMM(transD, transU, pobs)

    def scale_spectral_HMM(self, sHMM, tau):
        return sHMM.scale(tau)

    def lincomb_spectral_HMM(self, sHMM0: SpectralHMM, sHMM1: SpectralHMM, mu: float) -> SpectralHMM:
        return sHMM0.lincomb(sHMM1, mu)

    def setUp(self):
        self.transD0 = np.array([[1.0, 0, 0],
                                 [0, 0.1, 0],
                                 [0, 0, 0.1]])

        self.transU0 = np.array([[0.8, 0.1, 0.1],
                                 [-1, 1, 0],
                                 [1, 1, -2]])

        pobs = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1]
            ])
        self.pobs0 = pobs

        self.tau = 10
        self.transD1 = np.array([[1.0, 0, 0],
                                 [0, 0.9, 0],
                                 [0, 0, 0.8]])

        self.transU1 = np.fliplr(self.transU0)
        self.pobs1 = np.flipud(pobs)

        self.mu = 0.3
        self.shmm0 = self.create_spectral_HMM(self.transD0, self.transU0, self.pobs0)
        self.shmm1 = self.create_spectral_HMM(self.transD1, self.transU1, self.pobs1)

    def test_isdiagonal(self):
        self.assertTrue(self.shmm0.isdiagonal())
        self.assertTrue(self.shmm1.isdiagonal())
        # FIXME: NOT WORKING: self.assertFalse(self.shmm2.isdiagonal()) because the constructed matrix is
        # not a matrix of multinomials (some entries are negative, others are greater than 1)

    def test_lincomb(self):
        shmm_lc = self.lincomb_spectral_HMM(self.shmm0, self.shmm1, self.mu)
        self.assertTrue(shmm_lc.isdiagonal())
        # FIXME: more tests

    def test_scale(self):
        shmm0_scaled = self.scale_spectral_HMM(self.shmm0, self.tau)
        self.assertTrue(shmm0_scaled.isdiagonal())
        # FIXME: more tests

    def test_isclose(self):
        self.assertTrue(self.shmm0.isclose(self.shmm0))
