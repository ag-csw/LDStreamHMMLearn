import numpy as np
import scipy
from scipy import stats

from ldshmm.util.mm_family import MMFamily1
from ldshmm.util.quasi_mm import ConvexCombinationQuasiMM


class QMMFamily(object):
    """
    Family of QMMs
    """
    def sample(self, size=1):
        """
        sample routine to return an ndarray of SpectralMMs

        :param size: int (default=1) - size of the returned sample
        :return: ndarray instance of SpectralMM
       """
        raise NotImplementedError("Please implement this method")


class QMMFamily1(QMMFamily):
    """
    * No Dominant Relaxation Mode
    * No Dominant Metastable State
    """

    def __init__(self, mmfam,
                 edgewidth=1, edgeshift=0, gammamin=1, gammamax=1.01,
                 mu0 = lambda t: (np.tanh(t) + 1) / 2, delta=1):
        """

        :param mmfam: MMFamily
        :param edgewidth: (default=1) - base timescale of the phase transition
        :param edgeshift: (default=0) - base edge shift before multiplier
        :param gammamin: (default=1) - minimum value of edge shift multiplier
        :param gammamax: (default=1) - maximum value of edge shift multiplier
        :param mu0: (default=lambda t: (np.tanh(t) + 1) / 2) - template weight function
        :param delta: (default=1) -
        """

        self.mmfam = mmfam
        self.nstates = self.mmfam.nstates

        self.edgewidth = edgewidth
        self.edgeshift = edgeshift
        self.gammamin = gammamin
        self.gammamax = gammamax

        self.mu0 = mu0
        self.delta=delta
        self.gammadist = scipy.stats.uniform(self.gammamin, self.gammamax - self.gammamin)

    def _sample_one(self):
        """
        return a single sample QMM of the QMMFamily

        :return:
        """
        from ldshmm.util.mm_class import MMMScaled
        while True:
            try:
                # get two spectral MMs from the MM family self.mmfam
                mmms = self.mmfam.sample(2)
                # discard the sample if the basis vectors of the two MMs have
                # determinants of opposite sign, because there will always be
                # some convex combination of the bases that is singular, by continuity.
                # recalculate the endpoint mm using delta

                mmms0 = mmms[0]
                mmms1 = mmms[1]
                mmms_new = mmms0.lincomb(other=mmms1, mu=self.delta)
                mmms[1] = mmms_new

                if np.linalg.det(mmms[0].sMM.eigenvectors_left()) * np.linalg.det(mmms[1].sMM.eigenvectors_left()) < 0:
                    raise Exception
                gamma = self.gammadist.rvs(1)[0]
                self.gamma = gamma
                #print(self.edgeshift, gamma, self.edgewidth)

                # construct the base (taumeta = tauquasi = 1) weight function from the template
                def mu(t):
                    return self.mu0((t - self.edgeshift * gamma) / self.edgewidth)

                #print(self.edgeshift * gamma)

                # construct the convex combination quasi=stationary MM
                qmm = ConvexCombinationQuasiMM(mmms, mu)

                if self.delta != 0:
                    # Exclude most samples where some value of the convex combination
                    # fails to give a NSMM due to singularity of the convex combination
                    # of eigenvector matrices.

                    # take some NSMM from the scaled class
                    taumeta = 10
                    tauquasi = 10
                    mu0endpoint = 10
                    nsmm = qmm.eval(taumeta, tauquasi)

                    def f(x):
                        return np.linalg.det(nsmm.eval(x).transition_matrix)

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

                return qmm

            except Exception:
                continue

    def sample(self, size=1):
        mmms = np.empty(size, dtype=object)  # initialize sample vector
        for i in range(0, size):
            mmms[i] = self._sample_one()  # get one sample QMM of the QMMFamily
        return mmms
