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


    def test_convert_2d_to_list_of_rows(self):
        num_rows = 20
        array2d = np.reshape(np.arange(0,100),(num_rows, -1))

        list_of_rows = convert_2d_to_list_of_rows(array2d)

        assert len(array2d) == len(list_of_rows)
        assert len(array2d[0]) == len(list_of_rows[0])

    def test_sub_ndarray(self):
        num_rows = 20
        array2d = np.reshape(np.arange(0, 100), (num_rows, -1))

        sub_ndarray_size = 10
        sub_array2d = get_sub_ndarray(array2d, sub_ndarray_size)

        assert len(array2d[0]) == len(sub_array2d[0])
        for row in range(0, len(sub_array2d)):
            assert (array2d[row] == sub_array2d[row]).all()

    def test_reshape_trajs(self):
        num_rows = 20
        array2d = np.reshape(np.arange(0, 100), (num_rows, -1))
        num_trajectories2d, len_trajectory2d = np.shape(array2d)
        print(array2d)

        num_trajs_for_reshape = 5
        array3d = reshape_trajs(array2d, num_trajs_for_reshape)
        print(array3d)
        num_slices, num_trajectories3d, len_trajectory3d = np.shape(array3d)
        assert num_trajectories3d == num_trajs_for_reshape
        assert len_trajectory2d == len_trajectory3d