import logging

import numpy as np
import scipy
from scipy import stats
from scipy.stats import uniform

from ldshmm.util.spectral_mm import SpectralMM
from ldshmm.util.mm_class import MMMScaled
from msmtools.estimation import transition_matrix as _tm
from msmtools.analysis import is_transition_matrix as _is_tm
from pyemma.util.linalg import mdot

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
        trans = _tm(np.dot(transv, np.dot(transd, transu)))
        if _is_tm(trans) and self.is_scalable_tm(transd, transu, transv):
            return transd, transu, transv, trans
        else:
            # discard sample if trans is not a transition matrix or is not a scalable transition matrix
            return self.sample_transition_matrix()

    def _sample_one(self):
        transd, transu, transv, trans = self.sample_transition_matrix()
        smms = SpectralMM(transd, transu, transv, trans)  # construct a spectral MM
        try:
            return  MMMScaled(smms)
        except:
            return self._sample_one()

    def sample(self, size=1):
        mmms = np.empty(size, dtype=object) # initialize sample vector
        for i in range(0, size):
            mmms[i] = self._sample_one() # construct a spectral MM
        return mmms

    def is_scalable_tm(self, transd, transu, transv=None):
        if transv is None:
            transv = np.linalg.inv(transu)
        lntransd = np.diag(np.log(np.diag(transd)))
        delta = mdot(transv, lntransd, transu)
        deltadiag = np.diag(delta)
        deltatril = np.tril(delta, -1)
        deltatriu = np.triu(delta, 1)
        if np.all(deltadiag <= 0) and np.all(deltatril >= 0) and np.all(deltatriu >= 0):
            return True
        else:
            # For large scaling factors (tau), the scaling of the transition matrix approaches
            #
            #   I + (1/tau) ln( trans)
            #
            # This will be a  transition matrix for sufficiently large tau if
            #    1. all diagonal elements are <= 0
            #    2. all off-diagonal elements are >= 0
            #
            # Therefore the matrix is called "scalable" if it satisfies these properties.
            #
            # The diagonal decomposition is used for a fast calculation of the log
            return False
