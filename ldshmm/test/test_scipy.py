from unittest import TestCase
import numpy as np
import scipy.sparse
from time import process_time


class TestSpectralHMM(TestCase):

    def test(self):
        nstates = 4

        dtraj_0 = range(0,nstates)

        repeat = range(20, 50, 10)

        for j in repeat:
            dtraj = []
            for i in range(0,j):
                dtraj = np.concatenate((dtraj,
                                        dtraj_0))

            #print(dtraj)
            row = dtraj[0:-1]
            col = dtraj[1:]
            data = np.ones(row.size)

            t0 = process_time()
            C = scipy.sparse.coo_matrix((data, (row, col)), shape=(nstates, nstates))
            print("TIME: ",process_time()-t0)
            #print(C)
            print(C.toarray())
