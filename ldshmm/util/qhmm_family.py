import logging

import numpy as np
import scipy
from scipy import stats
from scipy.stats import uniform

from ldshmm.util.spectral_hmm import SpectralHMM


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
                 statconc=1):
        self.nstates = nstates
        self.nobserved = nobserved
        self.clusterconc = clusterconc
        self.withinclusterconc = withinclusterconc
        if clusters is None:
            self.clusters = clusters
            self.clusterconcvec = clusterconc * np.ones(nstates)
            self.cluster_rv = scipy.stats.dirichlet(self.clusterconcvec)
            logging.debug("Clusters are not specified. ")
        else:
            self.clusters = np.asarray(clusters)
            clustersizes = []
            clusterindices = []
            for i in range(0, self.nstates):
                indices = np.where(self.clusters == i)[0]
                clustersize = len(indices)
                clustersizes.append(clustersize)
                clusterindices.append(indices)
            self.clustersizes = clustersizes
            self.clusterindices = clusterindices
        self.eigenvaluemin = np.exp(-1.0)
        self.eigenvaluemax = np.exp(-1.0 / timescaledisp)
        self.dispscale = self.eigenvaluemax - self.eigenvaluemin
        self.eigenvaluedist = uniform(loc=self.eigenvaluemin, scale=self.dispscale)
        self.statconcvec = statconc * np.ones(nstates)
        self.stat_rv = scipy.stats.dirichlet(self.statconcvec)
        self.basisconcvec = np.ones(nstates)
        self.basis_rv = scipy.stats.dirichlet(self.basisconcvec)

    def sample_emission_matrix(self):
        # one sample
        if self.clusters is None:
            # sample for a crisp cluster assignment
            cluster_rvs = self.cluster_rv.rvs(self.nobserved)
            clusters = []
            for row in cluster_rvs:
                li = np.random.multinomial(1, row, 1)
                clusters.append(li[0].tolist().index(1))
        else:
            clusters = self.clusters
        clusters = np.asarray(clusters)
        logging.debug("Clusters: " + str(clusters))
        pobs = np.zeros((self.nstates, self.nobserved))
        for i in range(0, self.nstates):
            if self.clusters is None:
                indices = np.where(clusters == i)[0]
                clustersize = len(indices)
                if clustersize == 0:
                    return self.sample_emission_matrix()
            else:
                indices = self.clusterindices[i]
                clustersize = self.clustersizes[i]
            withinclusterconcvec = self.withinclusterconc * np.ones(clustersize)
            withincluster_rv = scipy.stats.dirichlet(withinclusterconcvec)
            pobs_sub = withincluster_rv.rvs(1)[0]
            pobs[i, indices] = pobs_sub
        return pobs

    def sample_eigenvalues(self):
        eigenvalues = np.ones((self.nstates), float)
        eigenvalues[1:] = self.eigenvaluedist.rvs(size=self.nstates - 1)
        return eigenvalues

    def sample_stationary(self):
        return self.stat_rv.rvs(1)

    def sample_basis(self):
        basis = np.empty((self.nstates, self.nstates))
        stat = self.sample_stationary()
        basis[0, :] = stat
        basis[1:, :] = self.basis_rv.rvs(self.nstates - 1) - stat
        if np.abs(np.linalg.det(basis)) > 1e-4:
            return basis
        else:
            return self.sample_basis()

    def sample_transition_matrix(self):
        transD = np.diag(self.sample_eigenvalues())
        transU = self.sample_basis()
        transV = np.linalg.inv(transU)
        trans = np.dot(transV, np.dot(transD, transU))
        if np.all(trans >= 0) and np.all(trans <= 1):
            return (transD, transU, transV, trans)
        else:
            return self.sample_transition_matrix()

    def sample(self, size = 1):
        smp = np.empty((size), object)
        for i in range(0, size):
          transd, transu, transv, trans = self.sample_transition_matrix()
          pobs = self.sample_emission_matrix()
          smp[i] = SpectralHMM(transd, transu, pobs, transv, pobs)
        return smp
