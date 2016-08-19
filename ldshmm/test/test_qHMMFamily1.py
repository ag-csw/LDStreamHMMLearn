from unittest import TestCase
import numpy as np
from qhmm_family import QHMMFamily1
import matplotlib.pyplot as plt

class TestQHMMFamily1(TestCase):
    def setUp(self):
        self.nstates = 4
        self.nobserved = 10;
        self.qmmf1_0 = QHMMFamily1(self.nstates, self.nobserved)
        self.clusters1 = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        self.clusters2 = [0, 1, 2, 1, 0, 1, 2, 0, 0, 1]
        self.qmmf1_1 = QHMMFamily1(nstates = self.nstates, nobserved = self.nobserved, clusters = self.clusters1)
        self.clusterconc = 3
        self.withinclusterconc = 0.5
        self.timescaledisp = 10
        self.statconc = 0.1
        self.edgewidth = 0.1
        self.edgeshift = 1
        self.gammamin = 1
        self.gammamax = 2
        self.qmmf1_2 = QHMMFamily1(self.nstates, self.nobserved, self.clusterconc, self.withinclusterconc,
                                   None, self.timescaledisp, self.statconc, self.edgewidth, self.edgeshift,
                                   self.gammamin, self.gammamax)


    def test_sample(self):
        for i in range(0, 100):
            qmm1_2 = self.qmmf1_2.sample()[0]
            if qmm1_2 is not None:
                nshmm1_2 = qmm1_2.eval(1, 1)

                def f(x):
                    return np.linalg.det(nshmm1_2.eval(x).transition_matrix)

                xvec = list(range(0, 10, 1))
                #print(xvec)
                yvec = list(map(f, xvec))
                #print(yvec)
                # plt.plot(xvec, yvec)
                # plt.ylabel('determinant')
                # plt.show()
                # plt.close()
            else:
                print('Sample Failed')
