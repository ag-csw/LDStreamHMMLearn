from unittest import TestCase
import pyemma.msm as MSM
import numpy as np

from ldshmm.util.spectral_hmm import SpectralHMM
from ldshmm.util.hmm_class import mHMMScaled


class TestMHMMScaled(TestCase):

    def setUp(self):
        self.transD0 = np.array([[1.0, 0, 0],
                                              [0, 0.2, 0],
                                              [0, 0, 0.1]])

        self.transU0 = np.array([[0.8, 0.1, 0.1],
                                             [-1, 1, 0],
                                             [1, 1, -2]])

        self.pobs0 = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1]
            ])

        self.tau = 10
        self.shmm0 = self.create_spectral_HMM(self.transD0, self.transU0, self.pobs0)
        self.mhmm = self.create_scaled_HMM_class(self.shmm0)

    ########################################
    # Spectral HMM Model Creation
    ########################################
    def create_spectral_HMM(self, transD, transU, pobs):
        return SpectralHMM(transD, transU, pobs)

    def scale_spectral_HMM(self, sHMM, tau):
        return sHMM.scale(tau)

    def lincomb_spectral_HMM(self, sHMM0: SpectralHMM, sHMM1: SpectralHMM, mu: float) -> SpectralHMM:
        return sHMM0.lincomb(sHMM1, mu)

    ########################################
    # Metastable HMM Class Creation
    ########################################
    def create_scaled_HMM_class(self, sHMM):
        return mHMMScaled(sHMM)

    def test_ismember(self):
        shmm0_scaled2 = self.mhmm.eval(self.tau)
        self.assertTrue(self.mhmm.ismember(shmm0_scaled2))

    def test_eval(self):
        shmm0_scaled = self.scale_spectral_HMM(self.shmm0, self.tau)
        shmm0_scaled2 = self.mhmm.eval(self.tau)
        self.assertTrue(np.allclose(shmm0_scaled.transition_matrix, shmm0_scaled2.transition_matrix))
        self.assertTrue(np.allclose(shmm0_scaled.observation_probabilities, shmm0_scaled2.observation_probabilities))
