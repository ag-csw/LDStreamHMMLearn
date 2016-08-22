from unittest import TestCase

import numpy as np
import scipy

from ldshmm.util.nonstationary_mm import ConvexCombinationNSMM
from ldshmm.util.spectral_mm import SpectralMM
# from ldshmm.util.qmm_family import QMMFamily1
from pyemma.util.linalg import mdot

class TestConvexCombinationNSMM(TestCase):
    def create_spectral_MM(self, transD, transU):
        return SpectralMM(transD, transU)

    def setUp(self):
        self.nstates = 3
        self.transD0 = np.array([[1.0, 0, 0],
                                 [0, 0.8, 0],
                                 [0, 0, 0.95]])

        self.transU0 = np.array([[0.4, 0.3, 0.3],
                                 [-1.0, 1.0, 0.0],
                                 [1.0, 1.0, -2.0]])


        self.tau = 10
        self.transD1 = np.array([[1.0, 0, 0],
                                 [0, 0.9, 0],
                                 [0, 0, 0.8]])

        self.transU1 = np.fliplr(self.transU0)

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

        self.smm0 = self.create_spectral_MM(self.transD0, self.transU0)
        self.smm1 = self.create_spectral_MM(self.transD1, self.transU3)
        self.smm2 = self.create_spectral_MM(self.transD1, self.transU4)

        self.lcnsmm0 = ConvexCombinationNSMM(self.smm0, self.smm1, mu0, self.timeendpoint)
        self.lcnsmm0_rev = ConvexCombinationNSMM(self.smm1, self.smm0, mu0, self.timeendpoint)
        self.lcnsmm1 = ConvexCombinationNSMM(self.smm0, self.smm1, mu1, 'infinity')
        self.lcnsmm2 = ConvexCombinationNSMM(self.smm0, self.smm0, mu1, 'infinity')
        self.lcnsmm3 = ConvexCombinationNSMM(self.smm0, self.smm2, mu1, 'infinity')

        self.trans2 = self.smm0.transition_matrix
        #print(self.transU0)
        self.p0_0 = self.transU0[0,:]
        #print(self.p0_0.dtype)
        #print(self.p0_0)
        self.p0_1 = self.transU1[0,:]

        #self.qmmf1_0 = QMMFamily1(self.nstates, self.nobserved)
        #self.nsmm1_0 = self.qmmf1_0.sample()[0].eval(1, self.edgewidth)
        #print("Starting Transition Matrix: ", self.nsmm1_0.eval(0).transition_matrix)
        #print("Ending Transition Matrix: ", self.nsmm1_0.eval(self.edgewidth*10).transition_matrix)

    def test_eval(self):
        self.assertTrue(self.lcnsmm0.eval(0).isclose(self.smm0))
        self.assertTrue(self.lcnsmm0.eval(self.timeendpoint).isclose(self.smm1))
        self.assertTrue(self.lcnsmm1.eval(10 * self.timeendpoint).isclose(self.smm1))

    def test_isclose(self):
        self.assertTrue(self.lcnsmm0.isclose(self.lcnsmm0))
        self.assertTrue(self.lcnsmm0.isclose(self.lcnsmm0, range(0, self.timeendpoint)))
        self.assertFalse(self.lcnsmm0.isclose(self.lcnsmm0_rev))

    def test_propagate(self):
        # If the nonstationary MM is actually stationary, should get the same result as eval(0).propagate
        print(self.lcnsmm2.propagate(self.p0_1, 0))
        print(self.lcnsmm2.propagate(self.p0_1, 1))
        print(self.lcnsmm2.propagate(self.p0_1, 2))
        print(self.lcnsmm2.propagate(self.p0_1, 3))
        print(self.lcnsmm2.eval(0).propagate(self.p0_1, 0))
        print(self.lcnsmm2.eval(0).propagate(self.p0_1, 1))
        print(self.lcnsmm2.eval(0).propagate(self.p0_1, 2))
        print(self.lcnsmm2.eval(0).propagate(self.p0_1, 3))
        print(self.p0_1)
        print(np.dot(self.p0_1, self.trans2))
        print(np.dot(self.p0_1, np.dot(self.trans2, self.trans2)))
        print(np.dot(self.p0_1, np.dot(self.trans2, np.dot(self.trans2, self.trans2))))
        print(mdot(self.p0_1, self.trans2, self.trans2, self.trans2))
        print(mdot(self.p0_1.T, self.trans2, self.trans2, self.trans2))
        print(self.lcnsmm2.eval(0).propagate(np.array([1.0, 0, 0]), 2))
        print(self.lcnsmm2.eval(0).propagate(np.array([0, 1.0, 0]), 2))
        print(self.lcnsmm2.eval(0).propagate(np.array([0, 0, 1.0]), 2))

        k = int(self.timeendpoint)
        self.assertTrue(
            np.allclose(self.lcnsmm2.propagate(self.p0_1, k), self.lcnsmm2.eval(0).propagate(self.p0_1, k)))

        # If MM0 and MM1 have the same stationary distribution, it should be stationary for the ccnsmm
        self.assertTrue(np.allclose(self.lcnsmm2.propagate(self.p0_0, k), self.p0_0))

    def test_simulate(self):
        N = int(self.timeendpoint)+1
        ntraj = 1000
        dtrajs = np.zeros((ntraj, N))
        for j in range(0, ntraj):
            dtrajs[j,:] = self.nsmm1_0.simulate(N)
        #print(htrajns)
        self.assertEqual(dtrajs[0].size, N)
        self.assertTrue(np.all(dtrajs < self.nstates))
        # FIXME: test if a lot of trajectories are sampled,
        # with start coming from some initial distribution
        # the distributions should look like the result of propagate
        for i in range(0, N):
            print("Sampled: ", scipy.stats.itemfreq(dtrajs[:, i])[:, 1]*(1/ntraj))
            print("Propagated: ", self.nsmm1_0.propagate(self.nsmm1_0.eval(0).stationary_distribution, i))
            self.assertTrue(
                np.allclose(scipy.stats.itemfreq(dtrajs[:, i])[:, 1]*(1/ntraj) ,
                self.nsmm1_0.propagate(self.nsmm1_0.eval(0).stationary_distribution, i),
                2e-1,
                1e-1
                ))

    def test_eigenvalues(self):
        print(self.transD0)
        print(self.smm0.eigenvalues())

    def test_eigenvectors(self):
        print(self.transU0)
        print(self.smm0.eigenvectors_left())

    def test_transition_matrix(self):
        print("Transition Matrix:\n")
        print(self.smm0.trans)
        print(mdot(self.smm0.eigenvectors_right(), np.diag(np.power(self.smm0.eigenvalues().real, 1)), self.smm0.eigenvectors_left()))
        print("Transition Matrix Squared:\n")
        print(mdot(self.smm0.trans, self.smm0.trans))
        print(mdot(self.smm0.eigenvectors_right(), np.diag(np.power(self.smm0.eigenvalues().real, 2)), self.smm0.eigenvectors_left()))
