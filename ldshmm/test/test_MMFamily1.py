from unittest import TestCase

import numpy as np

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.spectral_mm import SpectralMM
import math


class TestMM_Family1(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mmf1_1 = MMFamily1(nstates = self.nstates)
        self.timescaledisp = 10
        self.statconc = 0.1
        self.mmf1_2 = MMFamily1(self.nstates, self.timescaledisp, self.statconc)

    def test_sample_eigenvalues(self):
        eigenvalues = self.mmf1_0.sample_eigenvalues()
        print("Eigenvalues: ", eigenvalues)
        self.assertEqual(eigenvalues[0], 1.0)
        self.assertTrue(np.all(eigenvalues[1:] < 1))  # true for all real-diagonalizable MMs
        self.assertTrue(np.all(eigenvalues[1:] > 0))  # true for all real-diagonalizable MMs
        self.assertTrue(np.all(eigenvalues[1:] <= self.mmf1_0.eigenvaluemax))  # true for family 1
        self.assertTrue(np.all(eigenvalues[1:] >= self.mmf1_0.eigenvaluemin))  # true for family 1
        eigenvalues = self.mmf1_2.sample_eigenvalues()
        print("Eigenvalues: ", eigenvalues)
        self.assertEqual(eigenvalues[0], 1.0)
        self.assertTrue(np.all(eigenvalues[1:] < 1)) # true for all real-diagonalizable MMs
        self.assertTrue(np.all(eigenvalues[1:] > 0)) # true for all real-diagonalizable HMMs
        self.assertTrue(np.all(eigenvalues[1:] <= self.mmf1_2.eigenvaluemax)) # true for family 1
        self.assertTrue(np.all(eigenvalues[1:] >= self.mmf1_2.eigenvaluemin)) # true for family 1

    def test_sample_stationary(self):
        stat = self.mmf1_0.sample_stationary()
        self.assertTrue(np.allclose(np.sum(stat, 1), 1.0))
        print("Stationary Distribution: ", stat)
        stat = self.mmf1_1.sample_stationary()
        self.assertTrue(np.allclose(np.sum(stat, 1), 1.0))
        print("Stationary Distribution: ", stat)
        stat = self.mmf1_2.sample_stationary()
        self.assertTrue(np.allclose(np.sum(stat, 1), 1.0))
        print("Stationary Distribution: ", stat)

    def test_sample_basis(self):
        roweigenvectors0 = self.mmf1_0.sample_basis()
        print("Row Eigenvectors for 0: ", roweigenvectors0)
        coleigenvectors0 = np.linalg.inv(roweigenvectors0)
        print("Column Eigenvectors for 0: ", coleigenvectors0)

    def test_sample_transition_matrix(self):
        transD, transU, transV, trans = self.mmf1_0.sample_transition_matrix()
        print("Transition Matrix: ", trans)

    def test_sample(self):
        smm0 = self.mmf1_0.sample()[0]
        self.assertTrue(isinstance(smm0, SpectralMM))
        smm1 = self.mmf1_0.sample(2)[1]
        self.assertTrue(isinstance(smm1, SpectralMM))
        print("Determinant of Row Eigenvectors of Transition Matrix:", np.linalg.det(smm1.eigenvectors_left()))

    def test_sample_basis_fixed_statconc(self):
        self.nstates = 4
        self.statconc = 1/64
        self.mmf = MMFamily1(self.nstates, statconc=self.statconc)
        sample = self.mmf.sample_basis()
        print(sample)
        max_first_row = np.max(sample[0])
        print(max_first_row)
        print(abs(max_first_row-1))
        self.assertTrue(abs(max_first_row-1) <= 1e-4)

    def test_sample_basis_rows(self):
        sample = self.mmf1_0.sample_basis()
        print(sample)
        first_row = sample[0]
        sum_row = np.sum(first_row)
        print(sum_row)
        self.assertTrue(np.isclose(sum_row, 1.0))
        for i in sample[1:]:
            sum_row = np.sum(i)
            print(sum_row)
            self.assertTrue(np.isclose(sum_row, 0.0))
