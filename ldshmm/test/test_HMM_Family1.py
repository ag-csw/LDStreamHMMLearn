from unittest import TestCase
from hmm_family import HMM_Family1
import numpy as np
import scipy

class TestHMM_Family1(TestCase):
    def setUp(self):
        self.nstates = 2
        self.nobserved = 4;
        self.clusterconc = 1
        self.withinclusterconc = 1
        self.hmmf1_0 = HMM_Family1(self.nstates, self.nobserved, self.clusterconc, self.withinclusterconc)

    def test_sample_cluster(self):
        cl = self.hmmf1_0.sample_cluster()
        print(cl)
        clusters = []
        clustersizes = []
        pobs = np.zeros((self.nstates, self.nobserved))
        for i in range(0, self.nstates):
            indices = np.where(cl == i)[0]
            clustersize = len(indices)
            clusters.append(indices)
            clustersizes.append(clustersize)
            withinclusterconcvec = self.withinclusterconc * np.ones(clustersize)
            withincluster_rv = scipy.stats.dirichlet(withinclusterconcvec)
            pobs_sub = withincluster_rv.rvs(1)[0]
            pobs[i, indices] = pobs_sub
            print(pobs)
        print(clusters)
        print(clustersizes)
        self.assertTrue(True)
