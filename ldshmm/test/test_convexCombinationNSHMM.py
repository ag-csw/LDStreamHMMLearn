from unittest import TestCase

import numpy as np
import scipy

from ldshmm.util.nonstationary_hmm import ConvexCombinationNSHMM
from ldshmm.util.spectral_hmm import SpectralHMM
from hmm_family import HMMFamily1
from qhmm_family import QHMMFamily1

class TestConvexCombinationNSHMM(TestCase):
    def create_spectral_HMM(self, transD, transU, pobs):
        return SpectralHMM(transD, transU, pobs)

    def setUp(self):
        self.nhidden = 3
        self.nobserved = 4
        self.transD0 = np.array([[1.0, 0, 0],
                                 [0, 0.8, 0],
                                 [0, 0, 0.95]])

        self.transU0 = np.array([[0.4, 0.3, 0.3],
                                 [-1.0, 1.0, 0.0],
                                 [1.0, 1.0, -2.0]])
        #print(self.transU0.dtype)

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

        self.transU2 = np.array([[0.8, 0.1, 0.1],
                                 [-0.8, 1, -0.2],
                                 [1.1, 0.9, -2]])
        self.gamma = 0.5

        self.timeendpoint = 100

        def mu0(t):
            return t / 100

        self.edgewidth = self.timeendpoint/10

        def mu1(t):
            return (np.tanh(t / self.edgewidth - self.gamma) + 1) / 2

        eps = 0.1

        self.transU3 = (1-eps) * self.transU0 + eps * self.transU1
        self.transU4 = (1-eps) * self.transU0 + eps * self.transU2

        self.shmm0 = self.create_spectral_HMM(self.transD0, self.transU0, self.pobs0)
        self.shmm1 = self.create_spectral_HMM(self.transD1, self.transU3, self.pobs1)
        self.shmm2 = self.create_spectral_HMM(self.transD1, self.transU4, self.pobs1)

        self.lcnshmm0 = ConvexCombinationNSHMM(self.shmm0, self.shmm1, mu0, self.timeendpoint)
        self.lcnshmm0_rev = ConvexCombinationNSHMM(self.shmm1, self.shmm0, mu0, self.timeendpoint)
        self.lcnshmm1 = ConvexCombinationNSHMM(self.shmm0, self.shmm1, mu1, 'infinity')
        self.lcnshmm2 = ConvexCombinationNSHMM(self.shmm0, self.shmm0, mu1, 'infinity')
        self.lcnshmm3 = ConvexCombinationNSHMM(self.shmm0, self.shmm2, mu1, 'infinity')

        self.trans2 = self.shmm0.transition_matrix
        #print(self.transU0)
        self.p0_0 = self.transU0[0,:]
        #print(self.p0_0.dtype)
        #print(self.p0_0)
        self.p0_1 = self.transU1[0,:]

        hmm1_0 = HMMFamily1(self.nhidden, self.nobserved)
        self.qmmf1_0 = QHMMFamily1(hmm1_0)
        self.nshmm1_0 = self.qmmf1_0.sample()[0].eval(1, self.edgewidth)
        print("Starting Transition Matrix: ", self.nshmm1_0.eval(0).transition_matrix)
        print("Ending Transition Matrix: ", self.nshmm1_0.eval(self.edgewidth*10).transition_matrix)

    def test_eval(self):
        self.assertTrue(self.lcnshmm0.eval(0).isclose(self.shmm0))
        self.assertTrue(self.lcnshmm0.eval(self.timeendpoint).isclose(self.shmm1))
        self.assertTrue(self.lcnshmm1.eval(10 * self.timeendpoint).isclose(self.shmm1))

    def test_isclose(self):
        self.assertTrue(self.lcnshmm0.isclose(self.lcnshmm0))
        self.assertTrue(self.lcnshmm0.isclose(self.lcnshmm0, range(0, self.timeendpoint)))
        self.assertFalse(self.lcnshmm0.isclose(self.lcnshmm0_rev))

    def test_propagate(self):
        # If the nonstationary HMM is actually stationary, should get the same result as eval(0).propagate
        k = int(self.timeendpoint)
        #k=2
        #print(self.lcnshmm2.propagate(self.p0_1, k))
        #print(self.lcnshmm2.eval(0).propagate(self.p0_1, k))
        #print(np.dot(self.p0_1, np.dot(self.trans2, self.trans2)))
        #print(np.dot(self.p0_1, np.dot(self.trans2, np.dot(self.trans2, self.trans2))))
        self.assertTrue(
            np.allclose(self.lcnshmm2.propagate(self.p0_1, k), self.lcnshmm2.eval(0).propagate(self.p0_1, k).real))

        # If HMM0 and HMM1 have the same stationary distribution, it should be stationary for the ccnshmm
        self.assertTrue(np.allclose(self.lcnshmm2.propagate(self.p0_0, k), self.p0_0))

    def test_simulate(self):
        N = int(self.timeendpoint)+1
        ntraj = 1000
        htrajns = np.zeros((ntraj, N))
        otrajns = np.zeros((ntraj, N))
        for j in range(0, ntraj):
            trajns = self.nshmm1_0.simulate(N)
            #print(trajns)
            htrajns[j,:] = trajns[0]
            otrajns[j,:] = trajns[1]
        #print(htrajns)
        self.assertEqual(htrajns[0].size, N)
        self.assertTrue(np.all(htrajns < self.nhidden))
        self.assertTrue(np.all(otrajns < self.nobserved))
        # FIXME: test if a lot of trajectories are sampled,
        # with start coming from some initial distribution
        # the distributions should look like the result of propagate
        for i in range(0, N):
            print("Sampled: ", scipy.stats.itemfreq(htrajns[:, i])[:, 1]*(1/ntraj))
            print("Propagated: ", self.nshmm1_0.propagate(self.nshmm1_0.eval(0).stationary_distribution, i))
            self.assertTrue(
                np.allclose(scipy.stats.itemfreq(htrajns[:, i])[:, 1]*(1/ntraj) ,
                self.nshmm1_0.propagate(self.nshmm1_0.eval(0).stationary_distribution, i),
                2e-1,
                1e-1
                ))

