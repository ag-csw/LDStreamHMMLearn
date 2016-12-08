from unittest import TestCase
import numpy as np
from ldshmm.util.qmm_family import QMMFamily1
from ldshmm.util.mm_family import MMFamily1
import matplotlib.pyplot as plt

class TestQMMFamily1(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.qmmf1_0 = QMMFamily1(self.mmf1_0)
        self.timescaledisp = 10
        self.statconc = 0.1
        self.edgewidth = 0.1
        self.edgeshift = 10000
        self.gammamin = 1
        self.gammamax = 2
        self.mmf1_2 = MMFamily1(self.nstates,
                                   self.timescaledisp, self.statconc)
        self.qmmf1_2 = QMMFamily1(self.mmf1_2, self.edgewidth, self.edgeshift, self.gammamin, self.gammamax)
        self.numtestsamples = 10
        self.taumeta = 10
        self.tauquasi = 10

    def test_sample(self):
        for i in range(0, self.numtestsamples):
            print("Sample Number:", i)
            qmm1 = self.qmmf1_0.sample()[0]
            if qmm1 is not None:
                nsmm1 = qmm1.eval(self.taumeta, self.tauquasi)

                def f(x):
                    return np.linalg.det(nsmm1.eval(x).transition_matrix)

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
                nsmm1 = qmm1.eval(self.taumeta, self.tauquasi)

                def f(x):
                    return np.linalg.det(nsmm1.eval(x).transition_matrix)

                xvec = list(range(0, 2 * self.taumeta * self.tauquasi, ))
                yvec = list(map(f, xvec))
                # print("Minimum Determinant: ", min(map(np.absolute,yvec)))
                # if i == 0:
                #    plt.plot(xvec, yvec)
                #    plt.ylabel('determinant')
                #    plt.show
            else:
                print('Sample Failed')

    def test_mu(self):
        timepoints=np.arange(start=0, stop=3000, step=500)
        #for timepoint in timepoints:
        #    print(self.qmmf1_0.mu0(timepoint))
        model = self.qmmf1_0.sample()[0]

        assert model.eval(1).mu(0) == 0.5

        for timepoint in timepoints:
            print(timepoint, model.eval(1).mu(timepoint))
