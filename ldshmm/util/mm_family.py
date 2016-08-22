import logging

import numpy as np
import scipy
from scipy import stats
from scipy.stats import uniform

from ldshmm.util.spectral_mm import SpectralMM


class MMFamily(object):
    def sample(self, size=1):
        raise NotImplementedError("Please implement this method")


class MMFamily1(MMFamily):
    # No Dominant Relaxation Mode
    # No Dominant Metastable State
    # Crisply-clustered observables

    def __init__(self, nstates, timescaledisp=2, statconc=1):
        self.nstates = nstates # number of states
        self.eigenvaluemin = np.exp(-1.0)
        self.timescaledisp = timescaledisp # dispersion of the implied timescales in the base MM

        # Derived attributes
        self.eigenvaluemax = np.exp(-1.0 / self.timescaledisp)
        self.dispscale = self.eigenvaluemax - self.eigenvaluemin
        self.eigenvaluedist = uniform(loc=self.eigenvaluemin, scale=self.dispscale)
        self.statconcvec = statconc * np.ones(nstates)
        self.stat_rv = scipy.stats.dirichlet(self.statconcvec)
        self.basisconcvec = np.ones(nstates)
        self.basis_rv = scipy.stats.dirichlet(self.basisconcvec)

    def sample_eigenvalues(self):
        eigenvalues = np.ones(self.nstates, float) # initialize vector of eigenvalues
        eigenvalues[1:] = self.eigenvaluedist.rvs(size=self.nstates - 1) # sample for the non-stationary eigenvalues
        return eigenvalues

    def sample_stationary(self):
        return self.stat_rv.rvs(1) #sample for the stationary distribution from the initialized Dirichlet

    def sample_basis(self):
        basis = np.empty((self.nstates, self.nstates)) # initialize the left eigenvector matrix
        stat = self.sample_stationary()
        basis[0, :] = stat # stationary distribution is the left eigenvector with eigenvalue one
        basis[1:, :] = self.basis_rv.rvs(self.nstates - 1) - stat # other left eigenvectors have sum = 0
        if np.abs(np.linalg.det(basis)) > 1e-4:
            return basis
        else:
            return self.sample_basis() # discard sample if not linearly independent

    def sample_transition_matrix(self):
        transd = np.diag(self.sample_eigenvalues())
        transu = self.sample_basis()
        transv = np.linalg.inv(transu)
        trans = np.dot(transv, np.dot(transd, transu))
        if np.all(trans >= 0) and np.all(trans <= 1):
            return transd, transu, transv, trans
        else:
            return self.sample_transition_matrix() # discard sample if trans has elements that are not probabilities

    def sample(self, size=1):
        smms = np.empty(size, dtype=object) # initialize sample vector
        for i in range(0, size):
            transd, transu, transv, trans = self.sample_transition_matrix() # select a transmission matrix
            smms[i] = SpectralMM(transd, transu, transv, trans) # construct a spectral MM
        return smms
