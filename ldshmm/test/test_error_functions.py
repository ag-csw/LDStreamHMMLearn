from unittest import TestCase
from  ldshmm.util.error_functions import *
import numpy as np
from ldshmm.util.mm_family import MMFamily1


class TestSpectralHMM(TestCase):

    def setUp(self):
        nstates=4
        mmf1 = MMFamily1(nstates)
        self.model = mmf1.sample()[0]
        self.scaled_spectral_model = self.model.eval(4)
        self.estimated_trans = self.scaled_spectral_model.trans

    def test_transition_matrix_error(self):
        err = matrix_err(self.estimated_trans, self.scaled_spectral_model)
        assert err == 0


    def test_timescale_mean_rel_err(self):
        err = timescale_mean_rel_err(self.estimated_trans, self.scaled_spectral_model)
        assert err == 0


    def test_stat_dist_vec_err(self):
        err = stat_dist_vec_err(self.estimated_trans, self.scaled_spectral_model)
        assert err == 0