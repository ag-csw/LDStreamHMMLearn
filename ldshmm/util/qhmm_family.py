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

    def __init__(self, nstates, nobserved=None, clusterconc=1, withinclusterconc=1, clusters=None, timescaledisp=2,
                 statconc=1, edgewidth=1, edgeshift=1, gammamin=1, gammamax=10,
                 mu0 = lambda t: (np.tanh(t) + 1) / 2):
        self.nstates = nstates # number of hidden states
        # number of observed states
        if nobserved is None:
            if clusters is None:
                self.nobserved = nstates
            else:
                self.nobserved = len(clusters)
        else:
            self.nobserved = nobserved
        self.clusterconc = clusterconc # Dirchlet concentration of cluster assignment
        self.withinclusterconc = withinclusterconc # Dirchlet concentration within clusters
        self.clusters = clusters # crisp cluster assigment of observables to hidden variables
        self.timescaledisp = timescaledisp # dispersion of the implied timescales in the base HMMs
        self.statconc = statconc # Dirichlet concentration of stationary distribution
        self.edgewidth = edgewidth # base timescale of the phase transition
        self.edgeshift = edgeshift # base edge shift before multiplier
        self.gammamin = gammamin # minimum value of edge shift multiplier
        self.gammamax = gammamax # maximum value of edge shift multiplier
        # FIXME pass in the base weight function instead of all these parameters
        self.mu0 = mu0 # template weight function

        # FIXME pass in the HMMFamily instead of all these parameters
        self.hmmfam = HMMFamily1(self.nstates, self.nobserved, self.clusterconc, self.withinclusterconc, self.clusters,
                                 self.timescaledisp, self.statconc)
        self.gammadist = scipy.stats.uniform(self.gammamin, self.gammamax)

    def sample(self, size=1):
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

                # the template weight function
                # FIXME: pass this in as a parameter to the constructor
                def mu0(t):
                    return (np.tanh(t) + 1) / 2

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
                qhmm = self.sample()[0]

            smp[i] = qhmm
        return smp
