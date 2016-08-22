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
        self.qmmf1_1 = QHMMFamily1(nstates = self.nstates, clusters = self.clusters1)
        self.clusterconc = 3
        self.withinclusterconc = 0.5
        self.timescaledisp = 10
        self.statconc = 0.1
        self.edgewidth = 0.1
        self.edgeshift = 10
        self.gammamin = 1
        self.gammamax = 2
        self.qmmf1_2 = QHMMFamily1(self.nstates, self.nobserved, self.clusterconc, self.withinclusterconc,
                                   None, self.timescaledisp, self.statconc, self.edgewidth, self.edgeshift,
                                   self.gammamin, self.gammamax)
        # A (nonhidden) quasi-stationary Markov family
        self.clusters3 = [0, 1, 2, 3]
        self.qmmf1_3 = QHMMFamily1(nstates=self.nstates, clusters=self.clusters3)
        self.numtestsamples = 10
        self.taumeta = 10
        self.tauquasi = 10

    def test_sample(self):
        for i in range(0, self.numtestsamples):
            print("Sample Number:", i)
            qmm1 = self.qmmf1_0.sample()[0]
            if qmm1 is not None:
                nshmm1 = qmm1.eval(self.taumeta, self.tauquasi)

                def f(x):
                    return np.linalg.det(nshmm1.eval(x).transition_matrix)

                xvec = list(range(0, 2*self.taumeta* self.tauquasi, ))
                yvec = list(map(f, xvec))
                #print("Minimum Determinant: ", min(map(np.absolute, yvec)))
                # if i == 0:
                #    plt.plot(xvec, yvec)
                #    plt.ylabel('determinant')
                #    plt.show
            else:
                print('Sample Failed')
        for i in range(0, self.numtestsamples):
            print("Sample Number:", i)
            qmm1 = self.qmmf1_2.sample()[0]
            if qmm1 is not None:
                nshmm1 = qmm1.eval(self.taumeta, self.tauquasi)

                def f(x):
                    return np.linalg.det(nshmm1.eval(x).transition_matrix)

                xvec = list(range(0, 2 * self.taumeta * self.tauquasi, ))
                yvec = list(map(f, xvec))
                # print("Minimum Determinant: ", min(map(np.absolute,yvec)))
                # if i == 0:
                #    plt.plot(xvec, yvec)
                #    plt.ylabel('determinant')
                #    plt.show
            else:
                print('Sample Failed')
        for i in range(0, self.numtestsamples):
            print("Sample Number:", i)
            qmm1 = self.qmmf1_3.sample()[0]
            if qmm1 is not None:
                nshmm1 = qmm1.eval(self.taumeta, self.tauquasi)

                def f(x):
                    return np.linalg.det(nshmm1.eval(x).transition_matrix)

                xvec = list(range(0, 2*self.taumeta* self.tauquasi, ))
                yvec = list(map(f, xvec))
                #print("Minimum Determinant: ", min(map(np.absolute, yvec)))
                # if i == 0:
                #    plt.plot(xvec, yvec)
                #    plt.ylabel('determinant')
                #    plt.show
            else:
                print('Sample Failed')
        for i in range(0, self.numtestsamples):
            print("Sample Number:", i)
            qmm1 = self.qmmf1_2.sample()[0]
            if qmm1 is not None:
                nshmm1 = qmm1.eval(self.taumeta, self.tauquasi)

                def f(x):
                    return np.linalg.det(nshmm1.eval(x).transition_matrix)

                xvec = list(range(0, 2*self.taumeta* self.tauquasi, ))
                yvec = list(map(f, xvec))
                #print("Minimum Determinant: ", min(map(np.absolute,yvec)))
                #if i == 0:
                #    plt.plot(xvec, yvec)
                #    plt.ylabel('determinant')
                #    plt.show
            else:
                print('Sample Failed')
