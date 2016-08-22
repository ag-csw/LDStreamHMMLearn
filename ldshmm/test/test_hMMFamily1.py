from unittest import TestCase

import numpy as np

from hmm_family import HMMFamily1
from ldshmm.util.spectral_hmm import SpectralHMM


class TestHMM_Family1(TestCase):
    def setUp(self):
        self.nstates = 4
        self.nobserved = 10;
        self.hmmf1_0 = HMMFamily1(self.nstates, self.nobserved)
        self.clusters1 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        self.clusters2 = [0, 1, 2, 1, 0, 1, 2, 0, 0, 1]
        self.hmmf1_1 = HMMFamily1(nstates = self.nstates, nobserved = self.nobserved, clusters = self.clusters1)
        self.clusterconc = 3
        self.withinclusterconc = 0.5
        self.timescaledisp = 10
        self.statconc = 0.1
        self.hmmf1_2 = HMMFamily1(self.nstates, self.nobserved, self.clusterconc, self.withinclusterconc, None, self.timescaledisp, self.statconc)


    def test_sample_cluster(self):
        pobs = self.hmmf1_1.sample_emission_matrix()
        print("Emission Matrix: ", pobs)
        self.assertEqual(np.shape(pobs), (self.nstates, self.nobserved))
        self.assertTrue(np.allclose(np.sum(pobs, 1), 1.0))
        self.assertTrue(np.all(pobs >= 0), "Some element of the emission matrix is not nonnegative")
        self.assertTrue(np.all(pobs <= 1.0), "Some element of the emission matrix is not <= 1.0")
        for i in range(0, 1):
            pobs = self.hmmf1_0.sample_emission_matrix()
            # print("Emission Matrix: ", pobs)
            self.assertEqual(np.shape(pobs), (self.nstates, self.nobserved))
            self.assertTrue(np.allclose(np.sum(pobs, 1), 1.0))
            self.assertTrue(np.all(pobs >= 0), "Some element of the emission matrix is not nonnegative")
            self.assertTrue(np.all(pobs <= 1.0), "Some element of the emission matrix is not <= 1.0")

    def test_sample_eigenvalues(self):
        eigenvalues = self.hmmf1_0.sample_eigenvalues()
        print("Eigenvalues: ", eigenvalues)
        self.assertEqual(eigenvalues[0], 1.0)
        self.assertTrue(np.all(eigenvalues[1:] < 1))  # true for all real-diagonalizable HMMs
        self.assertTrue(np.all(eigenvalues[1:] > 0))  # true for all real-diagonalizable HMMs
        self.assertTrue(np.all(eigenvalues[1:] <= self.hmmf1_0.eigenvaluemax))  # true for family 1
        self.assertTrue(np.all(eigenvalues[1:] >= self.hmmf1_0.eigenvaluemin))  # true for family 1
        eigenvalues = self.hmmf1_2.sample_eigenvalues()
        print("Eigenvalues: ", eigenvalues)
        self.assertEqual(eigenvalues[0], 1.0)
        self.assertTrue(np.all(eigenvalues[1:] < 1)) # true for all real-diagonalizable HMMs
        self.assertTrue(np.all(eigenvalues[1:] > 0)) # true for all real-diagonalizable HMMs
        self.assertTrue(np.all(eigenvalues[1:] <= self.hmmf1_2.eigenvaluemax)) # true for family 1
        self.assertTrue(np.all(eigenvalues[1:] >= self.hmmf1_2.eigenvaluemin)) # true for family 1

    def test_sample_stationary(self):
        stat = self.hmmf1_0.sample_stationary()
        self.assertTrue(np.allclose(np.sum(stat, 1), 1.0))
        print("Stationary Distribution: ", stat)
        stat = self.hmmf1_1.sample_stationary()
        self.assertTrue(np.allclose(np.sum(stat, 1), 1.0))
        print("Stationary Distribution: ", stat)
        stat = self.hmmf1_2.sample_stationary()
        self.assertTrue(np.allclose(np.sum(stat, 1), 1.0))
        print("Stationary Distribution: ", stat)

    def test_sample_basis(self):
        roweigenvectors0 = self.hmmf1_0.sample_basis()
        print("Row Eigenvectors for 0: ", roweigenvectors0)
        coleigenvectors0 = np.linalg.inv(roweigenvectors0)
        print("Column Eigenvectors for 0: ", coleigenvectors0)

    def test_sample_transition_matrix(self):
        transD, transU, transV, trans = self.hmmf1_0.sample_transition_matrix()
        print("Transition Matrix: ", trans)

    def test_sample(self):
        shmm0 = self.hmmf1_0.sample()[0]
        self.assertTrue(isinstance(shmm0, SpectralHMM))
        shmm1 = self.hmmf1_0.sample(2)[1]
        self.assertTrue(isinstance(shmm1, SpectralHMM))
        print("Determinant of Row Eigenvectors of Transition Matrix:", np.linalg.det(shmm1.eigenvectors_left()))