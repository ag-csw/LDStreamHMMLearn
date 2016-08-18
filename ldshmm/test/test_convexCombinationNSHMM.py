from unittest import TestCase
import pyemma.msm as MSM
import numpy as np
from ldshmm.util.spectral_hmm import SpectralHMM
from ldshmm.util.nonstationary_hmm import ConvexCombinationNSHMM

class TestConvexCombinationNSHMM(TestCase):
    def create_spectral_HMM(self, transD, transU, pobs):
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
        def mu0(t):
            return t/100
        self.timeendpoint = 100

        def mu1(t):
            return (np.tanh(t-self.gamma)+1)/2

        self.timeendpoint = 100
        self.shmm0 = self.create_spectral_HMM(self.transD0, self.transU0, self.pobs0)
        self.shmm1 = self.create_spectral_HMM(self.transD1, self.transU1, self.pobs1)
        self.lcnshmm0 = ConvexCombinationNSHMM(self.shmm0, self.shmm1, mu0, self.timeendpoint)
        self.lcnshmm0_rev = ConvexCombinationNSHMM(self.shmm1, self.shmm0, mu0, self.timeendpoint)
        self.lcnshmm1 = ConvexCombinationNSHMM(self.shmm0, self.shmm1, mu1, 'infinity')

    def test_eval(self):
        self.assertTrue(self.lcnshmm0.eval(0).isclose(self.shmm0))
        self.assertTrue(self.lcnshmm0.eval(self.timeendpoint).isclose(self.shmm1))
        self.assertTrue(self.lcnshmm1.eval(100).isclose(self.shmm1))

    def test_isclose(self):
        self.assertTrue(self.lcnshmm0.isclose(self.lcnshmm0))
        self.assertTrue(self.lcnshmm0.isclose(self.lcnshmm0, range(0, self.timeendpoint)))
        self.assertFalse(self.lcnshmm0.isclose(self.lcnshmm0_rev))
