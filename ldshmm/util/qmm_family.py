import numpy as np
import scipy
from scipy import stats

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.quasi_mm import ConvexCombinationQuasiMM


class QMMFamily(object):
    def sample(self, size=1):
        raise NotImplementedError("Please implement this method")


class QMMFamily1(QMMFamily):
    # No Dominant Relaxation Mode
    # No Dominant Metastable State

    def __init__(self, mmfam,
                 edgewidth=1, edgeshift=0, gammamin=1, gammamax=1,
                 mu0 = lambda t: (np.tanh(t) + 1) / 2):
        self.mmfam = mmfam
        self.nstates = self.mmfam.nstates

        self.edgewidth = edgewidth # base timescale of the phase transition
        self.edgeshift = edgeshift # base edge shift before multiplier
        self.gammamin = gammamin # minimum value of edge shift multiplier
        self.gammamax = gammamax # maximum value of edge shift multiplier

        self.mu0 = mu0 # template weight function

        self.gammadist = scipy.stats.uniform(self.gammamin, self.gammamax)

    def _sample_one(self):
        try:
            # get two spectral MMs from the MM family self.mmfam
            mmms = self.mmfam.sample(2)
            # discard the sample if the basis vectors of the two MMs have
            # determinants of opposite sign, because there will always be
            # some convex combination of the bases that is singular, by continuity.
            if np.linalg.det(mmms[0].sMM.eigenvectors_left()) * np.linalg.det(mmms[1].sMM.eigenvectors_left()) < 0:
                raise Exception
            gamma = self.gammadist.rvs(1)

            # construct the base (taumeta = tauquasi = 1) weight function from the template
            def mu(t):
                return self.mu0((t - self.edgeshift * gamma) / self.edgewidth)

            # construct the convex combination quasi=stationary MM
            try:
               qmm = ConvexCombinationQuasiMM(mmms, mu)
            except:
                raise Exception

            # Exclude most samples where some value of the convex combination
            # fails to give a NSMM due to singularity of the convex combination
            # of eigenvector matrices.

            # take some NSMM from the scaled class
            taumeta = 10
            tauquasi = 10
            mu0endpoint = 10
            try:
                nsmm = qmm.eval(taumeta, tauquasi)
            except:
                raise Exception

            def f(x):
                return np.linalg.det(nsmm.eval(x).transition_matrix)

            # FIXME: a smarter way to test this might be to find the minimum
            # of the function
            #
            # S(x) = sgn(f(0)) * f(t)
            #
            # If it is negative, then discard the sample
            xvec = list(range(0, taumeta * tauquasi * int(self.edgewidth * mu0endpoint + self.edgeshift * gamma)))
            try:
                yvec = list(map(f, xvec))
            except:
                raise Exception
            if len([y for y in yvec if y > 0]) * len([y for y in yvec if y < 0]) > 0:
                raise Exception
            return qmm

        except Exception:
            #return None
            return self._sample_one()

    def sample(self, size=1):
        mmms = np.empty(size, dtype=object)  # initialize sample vector
        for i in range(0, size):
            mmms[i] = self._sample_one()  # construct a spectral MM
        return mmms