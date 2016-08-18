import scipy
import numpy as np

class HMM_Family(object):


    def sample(self, size=1):
        raise NotImplementedError("Please implement this method")

class HMM_Family1(HMM_Family):
    # No Dominant Relaxation Mode
    # No Dominant Metastable State
    # Crisply-clustered observables
    # Initial Distribution is the Stationary Distributino

    def __init__(self, nstates, nobserved, clusterconc, withinclusterconc):
        self.nstates = nstates
        self.nobserved = nobserved
        self.clusterconc = clusterconc
        self.withinclusterconc = withinclusterconc
        self.clusterconcvec = clusterconc * np.ones(nstates)
        self.cluster_rv = scipy.stats.dirichlet(self.clusterconcvec)

    def sample_cluster(self, size = 1):
        # one sample
        cluster_rvs = self.cluster_rv.rvs(self.nobserved)
        cl = []
        for row in cluster_rvs:
            li = np.random.multinomial(1, row, 1)
            ci = li[0].tolist().index(1)
            cl.append(li[0].tolist().index(1))
        return np.asarray(cl)

    def sample(self, size = 1):
        pass