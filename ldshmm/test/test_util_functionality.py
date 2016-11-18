from unittest import TestCase
from ldshmm.util.mm_family import MMFamily1
from msmtools.estimation import transition_matrix as _tm
from ldshmm.util.util_functionality import *
import numpy as np
import math

class Test_QMM_MM(TestCase):

    def setUp(self):
        self.numstates = 4
        self.mm = MMFamily1(nstates=self.numstates)
        self.mm_sampled = self.mm.sample()[0]

    def test_basics(self):

        self.numtrajs = 1
        self.simulation = simulate_and_store(self.mm_sampled, num_trajs_simulated=self.numtrajs)
        # check if nu        import mathmtrajs num trajectories were simulated
        for key, value in self.simulation.items():
            shape = np.shape(value)[0]
            assert shape == self.numtrajs

        # check if all elements are in the correct range of (0, numstates)
        for key, value in self.simulation.items():
            for item in value:
                for state in item:
                    assert state <= self.numstates and  state >= 0

        # make sure that the array is full of simulated data, and not just zeros
        end_states = []
        for key, value in self.simulation.items():
            end_states.append(value[-1])
        assert np.max(end_states) > 0


    def test_close(self):
        self.numtrajs = int(math.pow(2,12))
        self.simulation = simulate_and_store(self.mm_sampled, num_trajs_simulated=self.numtrajs, len_trajectory=2)

        taumeta = 8
        simulated_data_taumeta = self.simulation[taumeta]
        mm_spectral = self.mm_sampled.eval(taumeta)

        count_matrix = estimate_via_sliding_windows(data=simulated_data_taumeta, num_states=self.numstates, initial=True)
        normalized_count_matrix = _tm(count_matrix)
        print(normalized_count_matrix)
        print(mm_spectral.trans)
        assert np.allclose(a=normalized_count_matrix, b=mm_spectral.trans, rtol=1e-1, atol=1e-1)


    def test_subsampling_close(self):
        self.numtrajs = int(math.pow(2, 12))
        self.simulation = simulate_and_store(self.mm_sampled, num_trajs_simulated=self.numtrajs, len_trajectory=3)

        taumeta = 4
        simulated_data_taumeta = self.simulation[taumeta]
        mm_spectral = self.mm_sampled.eval(taumeta)

        count_matrix = estimate_via_sliding_windows(data=simulated_data_taumeta, num_states=self.numstates,
                                                    initial=True, lag=2)
        normalized_count_matrix = _tm(count_matrix)
        print(normalized_count_matrix)
        print(mm_spectral.trans)
        assert np.allclose(a=normalized_count_matrix, b=mm_spectral.trans, rtol=1e-1, atol=1e-1)