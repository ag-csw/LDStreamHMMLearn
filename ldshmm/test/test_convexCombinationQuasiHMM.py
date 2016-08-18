from unittest import TestCase
import numpy as np
from ldshmm.util.spectral_hmm import SpectralHMM
from ldshmm.util.quasi_hmm import ConvexCombinationQuasiHMM


class TestConvexCombinationQuasiHMM(TestCase):
    @staticmethod
    def create_spectral_HMM(transD, transU, pobs):
        return SpectralHMM(transD, transU, pobs)

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

        self.gamma = 0.5

        def mu(t):
            return t / 100

        self.timeendpoint = 100
        self.shmm0 = self.create_spectral_HMM(self.transD0, self.transU0, self.pobs0)
        self.shmm1 = self.create_spectral_HMM(self.transD1, self.transU1, self.pobs1)
        self.qhmm = ConvexCombinationQuasiHMM(self.shmm0, self.shmm1, mu, self.timeendpoint)

    def test_eval(self):
        shmm0_test = self.qhmm.eval(1, 1).eval(0)
        self.assertTrue(self.shmm0.isclose(shmm0_test))
