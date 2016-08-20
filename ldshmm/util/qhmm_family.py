import numpy as np
import scipy
from scipy import stats

from ldshmm.util.hmm_family import HMMFamily1
from ldshmm.util.quasi_hmm import ConvexCombinationQuasiHMM


class QHMMFamily(object):
    def sample(self, size=1):
        raise NotImplementedError("Please implement this method")


class QHMMFamily1(QHMMFamily):
    # No Dominant Relaxation Mode
    # No Dominant Metastable State
    # Crisply-clustered observables
    # Initial Distribution is the Stationary Distribution
    # tanh weight function

    def __init__(self, nstates, nobserved, clusterconc=1, withinclusterconc=1, clusters=None, timescaledisp=2,
                 statconc=1, edgewidth=1, edgeshift=1, gammamin=1, gammamax=10):
        self.nstates = nstates
        self.nobserved = nobserved
        self.clusterconc = clusterconc
        self.withinclusterconc = withinclusterconc
        self.clusters = clusters
        self.timescaledisp = timescaledisp
        self.statconc = statconc
        self.edgewidth = edgewidth
        self.edgeshift = edgeshift
        self.gammamin = gammamin
        self.gammamax = gammamax
        self.hmmfam = HMMFamily1(self.nstates, self.nobserved, self.clusterconc, self.withinclusterconc, self.clusters,
                                 self.timescaledisp, self.statconc)
        self.gammadist = scipy.stats.uniform(self.gammamin, self.gammamax)



    def sample(self, size=1):
        smp = np.empty(size, object)
        for i in range(0, size):
            try:
                shmms = self.hmmfam.sample(2)
                gamma = self.gammadist.rvs(1)

                def mu0(t):
                    return (np.tanh(t) + 1) / 2

                def mu(t):
                    return mu0( (t - self.edgeshift * gamma) / self.edgewidth)

                qhmm = ConvexCombinationQuasiHMM(shmms, mu)
                nshmm = qhmm.eval(1, 1)

                def f(x):
                    return np.linalg.det(nshmm.eval(x).transition_matrix)

                xvec = list(range(0, int(self.edgewidth*10 + self.edgeshift*gamma)))

                list(map(f, xvec))

            except Exception:
                qhmm = self.sample()[0]

            smp[i] = qhmm
        return smp
