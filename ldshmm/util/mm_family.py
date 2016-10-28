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


    timescale_max = None

    def __init__(self, nstates, timescaledisp=2, statconc=1, timescale_min=1):
        """

        :param nstates: int - number of states
        :param timescaledisp: (default=2) dispersion of the implied timescales in the base MM
        :param statconc: (default=1) state concentration used to sample for the stationary distribution from the initialized Dirichlet
        :param timescale_min: (default=1) minimum of the implied timescales in the base MM
        """

        self.nstates = nstates
        self.timescale_min = timescale_min
        self.timescaledisp = timescaledisp
        self.timescale_max = self.timescaledisp * self.timescale_min
        self.eigenvaluemin = np.exp(-1.0 / self.timescale_min)

        # Derived attributes
        self.eigenvaluemax = np.exp(-1.0 / (self.timescaledisp * self.timescale_min))
        self.dispscale = self.eigenvaluemax - self.eigenvaluemin
        self.eigenvaluedist = uniform(loc=self.eigenvaluemin, scale=self.dispscale)
        self.statconcvec = statconc * np.ones(nstates)
        self.stat_rv = scipy.stats.dirichlet(self.statconcvec)
        self.basisconcvec = np.ones(nstates)
        self.basis_rv = scipy.stats.dirichlet(self.basisconcvec)

    def sample_eigenvalues(self):
        """
        sample for the non-stationary eigenvalues

        :return: ndarray of eigenvalues
        """

        eigenvalues = np.ones(self.nstates, float) # initialize vector of eigenvalues
        eigenvalues[1:] = self.eigenvaluedist.rvs(size=self.nstates - 1)
        return eigenvalues

    def sample_stationary(self):
        """
        sample for the stationary distribution from the initialized Dirichlet

        :return: ndarray of stationary distribution
        """

        return self.stat_rv.rvs(1)

    def sample_basis(self):
        """
        sample basis (left eigenvector matrix)
        ToDo Document

        :return: ndarray of row eigenvectors
        """

        basis = np.empty((self.nstates, self.nstates)) # initialize the left eigenvector matrix
        stat = self.sample_stationary()
        basis[0, :] = stat # stationary distribution is the left eigenvector with eigenvalue one
        basis[1:, :] = self.basis_rv.rvs(self.nstates - 1) - stat # other left eigenvectors have sum = 0
        if np.abs(np.linalg.det(basis)) > 1e-4:
            return basis
        else:
            return self.sample_basis() # discard sample if not linearly independent

    def sample_transition_matrix(self):
        """
        sample transition matrix by calculating D (transd), U (transu) and V (transv)

        :return: transd - diagonal array, transu - left eigenvector matrix, transv - inverse matrix of transu, trans - dot product transd * transu * transv
        """

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
        """
        sample routine to return a MMMScaled

        :return: MMMScaled which is an instance of MMMScaled
        """

        transd, transu, transv, trans = self.sample_transition_matrix()
        smms = SpectralMM(transd, transu, transv, trans)  # construct a spectral MM
        try:
            return  MMMScaled(smms)
        except:
            return self._sample_one()

    def sample(self, size=1):
        """
       sample routine to return an ndarray of MMMScaled

       :param size: int (default=1) - size of the returned sample
       :return: ndarray instance of MMMScaled
       """

        mmms = np.empty(size, dtype=object) # initialize sample vector
        for i in range(0, size):
            mmms[i] = self._sample_one() # construct a spectral MM
        return mmms

    def is_scalable_tm(self, transd, transu, transv=None):
        """
        ToDo Document

        :param transd: ndarray - diagonal array
        :param transu: ndarray - left eigenvector matrix
        :param transv: ndarray (default=None) - inverse matrix of transu
        :return: bool - True if D*U*V is scalable, otherwise False
        """

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
