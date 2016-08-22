import logging

import numpy as np
import scipy
from scipy import stats
from scipy.stats import uniform

from ldshmm.util.spectral_hmm import SpectralHMM


class HMMFamily(object):
    def sample(self, size=1):
        raise NotImplementedError("Please implement this method")


class HMMFamily1(HMMFamily):
    # No Dominant Relaxation Mode
    # No Dominant Metastable State
    # Crisply-clustered observables

    def __init__(self, nstates, nobserved=None, clusterconc=1, withinclusterconc=1, clusters=None, timescaledisp=2,
                 statconc=1):
        self.nstates = nstates # number of hidden states
        # number of observed states
        if self.nobserved is None:
            self.nobserved = self.nstates
        else:
            self.nobserved = nobserved
        self.clusterconc = clusterconc # Dirchlet concentration of cluster assignment
        self.withinclusterconc = withinclusterconc # Dirchlet concentration within clusters
        # crisp cluster assigment of observables to hidden variables
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
                if clustersize > 0:
                    clustersizes.append(clustersize)
                else:
                    # FIXME This could be extended to allow submodels
                    raise AssertionError("Empty clusters are not allowed.")
                clusterindices.append(indices)
            self.clustersizes = clustersizes
            self.clusterindices = clusterindices
        self.eigenvaluemin = np.exp(-1.0)
        self.timescaledisp = timescaledisp # dispersion of the implied timescales in the base HMM

        # Derived attributes
        self.eigenvaluemax = np.exp(-1.0 / self.timescaledisp)
        self.dispscale = self.eigenvaluemax - self.eigenvaluemin
        self.eigenvaluedist = uniform(loc=self.eigenvaluemin, scale=self.dispscale)
        self.statconcvec = statconc * np.ones(nstates)
        self.stat_rv = scipy.stats.dirichlet(self.statconcvec)
        self.basisconcvec = np.ones(nstates)
        self.basis_rv = scipy.stats.dirichlet(self.basisconcvec)

    def sample_emission_matrix(self):
        # one sample only
        if self.clusters is None:
            # sample for a crisp cluster assignment
            cluster_rvs = self.cluster_rv.rvs(self.nobserved) # a multinomial sample from the Dirichlet for each observed state
            clusters = []
            for row in cluster_rvs:
                li = np.random.multinomial(1, row, 1) # sample to get the cluster assignment
                clusters.append(li[0].tolist().index(1))
        else:
            clusters = self.clusters
        clusters = np.asarray(clusters)
        logging.debug("Clusters: " + str(clusters))

        pobs = np.zeros((self.nstates, self.nobserved)) #initialize the emission matrix to zeros
        for i in range(0, self.nstates):
            if self.clusters is None:
                # calculate cluster indices: observable indices for each hidden state cluster
                indices = np.where(clusters == i)[0]
                clustersize = len(indices)
                if clustersize == 0:
                    return self.sample_emission_matrix() # discard if any cluster is empty
            else:
                # use pre-calculated cluster indices
                indices = self.clusterindices[i]
                clustersize = self.clustersizes[i]
            withinclusterconcvec = self.withinclusterconc * np.ones(clustersize)
            withincluster_rv = scipy.stats.dirichlet(withinclusterconcvec) # sample for within-cluster emission
            pobs[i, indices] = withincluster_rv.rvs(1)[0] # set probabilities in emission matrix
        return pobs

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
        shmms = np.empty(size, dtype=object) # initialize sample vector
        for i in range(0, size):
            transd, transu, transv, trans = self.sample_transition_matrix() # select a transmission matrix
            pobs = self.sample_emission_matrix() # select an emission matrix
            shmms[i] = SpectralHMM(transd, transu, pobs, transv, trans) # construct a spectral HMM
        return shmms
