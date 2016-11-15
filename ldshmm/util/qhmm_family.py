import numpy as np
import scipy
from scipy import stats

from ldshmm.util.hmm_family import HMMFamily1
from ldshmm.util.quasi_hmm import ConvexCombinationQuasiHMM


class QHMMFamily(object):
    """
   Family of QMMs
   """

    def sample(self, size=1):
        """
        sample routine to return an ndarray of SpectralHMMs

        :param size: int (default=1) - size of the returned sample
        :return: ndarray instance of SpectralHMM
        """
        raise NotImplementedError("Please implement this method")


class QHMMFamily1(QHMMFamily):
    """
    * No Dominant Relaxation Mode
    * No Dominant Metastable State
    * Crisply-clustered observables
    """

    def __init__(self, hmmfam,
                 edgewidth=1, edgeshift=0, gammamin=1, gammamax=1,
                 mu0 = lambda t: (np.tanh(t) + 1) / 2):
        """

        :param hmmfam: HMMFamily
        :param edgewidth: (default=1) - base timescale of the phase transition
        :param edgeshift: (default=0) - base edge shift before multiplier
        :param gammamin: (default=1) - minimum value of edge shift multiplier
        :param gammamax: (default=1) - maximum value of edge shift multiplier
        :param mu0: (default=lambda t: (np.tanh(t) + 1) / 2) - template weight function
        """

        self.hmmfam = hmmfam
        self.nstates = self.hmmfam.nstates
        self.nobserved = self.hmmfam.nobserved

        self.edgewidth = edgewidth
        self.edgeshift = edgeshift
        self.gammamin = gammamin
        self.gammamax = gammamax

        self.mu0 = mu0
        self.gammadist = scipy.stats.uniform(self.gammamin, self.gammamax)

    def sample(self, size=1):
        """
        ToDo Document

        :return:
        """

        smp = np.empty(size, object)
        for i in range(0, size):
            try:
                # get two spectral HMMs from the HMM family self.hmmfam
                shmms = self.hmmfam.sample(2)
                # discard the sample if the basis vectors of the two HMMs have
                # determinants of opposite sign, because there will always be
                # some convex combination of the bases that is singular, by continuity.
                if np.linalg.det(shmms[0].eigenvectors_left()) * np.linalg.det(shmms[1].eigenvectors_left()) < 0:
                    raise Exception
                gamma = self.gammadist.rvs(1)

                # construct the base (taumeta = tauquasi = 1) weight function from the template
                def mu(t):
                    return self.mu0((t - self.edgeshift * gamma) / self.edgewidth)

                # construct the convex combination quasi=stationary HMM
                qhmm = ConvexCombinationQuasiHMM(shmms, mu)

                # Exclude most samples where some value of the convex combination
                # fails to give a NSHMM due to singularity of the convex combination
                # of eigenvector matrices.

                # take some NSHMM from the scaled class
                taumeta = 10
                tauquasi = 10
                mu0endpoint = 10
                nshmm = qhmm.eval(taumeta, tauquasi)

                def f(x):
                    return np.linalg.det(nshmm.eval(x).transition_matrix)

                # FIXME: a smarter way to test this might be to find the minimum
                # of the function
                #
                # S(x) = sgn(f(0)) * f(t)
                #
                # If it is negative, then discard the sample
                xvec = list(range(0, taumeta * tauquasi * int(self.edgewidth * mu0endpoint + self.edgeshift * gamma)))
                yvec = list(map(f, xvec))
                if len([y for y in yvec if y > 0]) * len([y for y in yvec if y < 0]) > 0:
                    raise Exception

            except Exception:
                # FIXME avoid recursion
                qhmm = self.sample()[0]

            smp[i] = qhmm
        return smp
