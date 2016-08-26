from unittest import TestCase
import numpy as np
import pyemma.msm as MSM
import pyemma.msm.estimators as _MME
from msmtools.estimation import transition_matrix as _tm
from msmtools.estimation.sparse.count_matrix import count_matrix_coo2_mult
from time import process_time

from mm_family import MMFamily1

class Approach_Test(TestCase):
    def setUp(self):
        self.nstates = 4
        self.mmf1_0 = MMFamily1(self.nstates)
        self.mm1_0_0 = self.mmf1_0.sample()[0]
        print("Base Transition Matrix: \n", self.mm1_0_0.transition_matrix)

        self.taumeta = 3
        self.tauquasi = 3
        self.nstep = 10 * self.taumeta * self.tauquasi
        self.nwindow = 1000 * self.nstep
        self.numsteps = 1
        self.lentraj = self.nwindow + self.numsteps * self.nstep + 1
        self.ntraj = 1
        self.r = (self.nwindow - self.nstep) / self.nwindow
        self.mm1_0_0_scaled = self.mm1_0_0.scale(self.taumeta)
        self.data1_0_0 = []
        for i in range(0, self.ntraj):
            self.data1_0_0.append(self.mm1_0_0_scaled.simulate(self.lentraj))
        #print(self.data1_0_0)

    def estimate_via_sliding_windows(self, data):
        C = count_matrix_coo2_mult(data, lag=1, sliding=False, sparse=False, nstates = self.nstates)

        #import msmtools.estimation as msmest
        #Csparse = msmest.effective_count_matrix(dataslice, 2)
        #Csub = Csparse.toarray()
        #cshape = Csub.shape
        #C = np.zeros((self.nstates, self.nstates), dtype=float)
        #C[0:cshape[0], 0:cshape[1]] = Csub
        #print("Csparse:\n", Csparse)
        #print("Csub:\n", Csparse)
        print("C:\n", C)
        return C

    def test_approach(self):
        dataarray = np.asarray(self.data1_0_0)
        k=0 # zero-th window, of length nwindow
        data0 = dataarray[:, k*self.nstep : (self.nwindow + k*self.nstep)]
        dataslice0 = []
        for i in range(0, self.ntraj):
            dataslice0.append(data0[i,:])
        C0 = self.estimate_via_sliding_windows(dataslice0) # count matrix for whole window
        print("C0 :\n", C0)
        print()
        A0 = _tm(C0)
        print("A0 :\n", A0)
        print()

        k=1 # first window, after slide of nstep
        data1 = dataarray[:, k*self.nstep : (self.nwindow + k*self.nstep)]
        dataslice1 = []
        for i in range(0, self.ntraj):
            dataslice1.append(data1[i,:])
        t0 = process_time()
        C1 = self.estimate_via_sliding_windows(dataslice1) # count matrix for whole window
        #print("C1 :\n", C1)
        #print()
        A1 = _tm(C1)
        etime0 = process_time()-t0
        print("A1 :\n", A1)
        print()

        data1new = dataarray[:, self.nwindow - 1 : (self.nwindow + 1*self.nstep)]
        dataslice1new = []
        for i in range(0, self.ntraj):
            dataslice1new.append(data1new[i,:])
        t1 = process_time()
        C1new = self.estimate_via_sliding_windows(dataslice1new) # count matrix for just new transitions
        #print("C1new :\n", C1new)
        #print("Total for C1new:", np.sum(C1new))
        #print()
        #weight0 = (self.r/(1.0-self.r)) / np.sum(C0)
        #weight1 = self.nstep / np.sum(C1new)
        weight0 = self.r
        weight1 = 1.0
        #print("Weights : ", weight0, weight1)
        C1bayes = weight0 * C0 + weight1 * C1new
        #print("C1bayes :\n", C1bayes)
        #print("Total for C1bayes:", np.sum(C1bayes))
        #print()
        A1bayes = _tm(C1bayes)
        etime1 = process_time()-t1
        print("A1bayes :\n", A1bayes)
        print()
        print("Real A :\n", self.mm1_0_0_scaled.trans)
        print("Error with Full Window : ", np.linalg.norm(A1 - self.mm1_0_0_scaled.trans))
        print("Error with Prior : ", np.linalg.norm(A1bayes - self.mm1_0_0_scaled.trans))
        print(self.mm1_0_0_scaled.eigenvalues())
        print("Times:", etime0, etime1)