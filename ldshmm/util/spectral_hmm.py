"""
This class is for spectral HMMs. which are HMMs specified in terms of a particular Jordan decomposition of the
transmission matrix.
"""
import numpy as np
from pyemma.msm.models.hmsm import HMSM as _HMM


class SpectralHMM(_HMM):
    def __init__(self, transd, transu, pobs, transv=None, trans=None):
        assert len(transd.shape) is 2, "transd is not a matrix"
        assert len(transu.shape) is 2, "transu is not a matrix"
        assert transd.shape[0] is transd.shape[1], "transd is not square"
        assert transu.shape[0] is transu.shape[1], "transu is not square"
        assert transu.shape[0] is transd.shape[0], "transd and transu do not have the same number of rows"
        self.transD = transd
        self.transU = transu
        assert not np.linalg.det(transu) == 0.0, "transu is not invertible"
        if transv is None:
            self.transV = np.linalg.inv(self.transU)
        else:
            self.transV = transv
        if trans is None:
            self.trans = np.dot(np.dot(transv, transd), transu)
        else:
            self.trans = trans
        super(SpectralHMM, self).__init__(self.trans, pobs, transu[0], '1 step')

    def isdiagonal(self):
        return np.allclose(self.transD, np.diag(np.diag(self.transD)))

    def lincomb(self, other, mu):
        assert -1e-8 <= mu <= 1 + 1e-8, "weight is not between 0 and 1, inclusive"
        assert self.isdiagonal(), "self is not diagonal"
        assert other.isdiagonal(), "other is not diagonal"
        transd = (1.0 - mu) * self.transD + mu * other.transD
        transu = (1.0 - mu) * self.transU + mu * other.transU
        pobs = (1.0 - mu) * self.pobs + mu * other.pobs
        return SpectralHMM(transd, transu, pobs)

    def scale(self, tau):
        assert tau > 0, "scaling factor is not positive"
        assert self.isdiagonal(), "self is not diagonal"
        transd_scaled = np.diag(np.power(np.diag(self.transD), 1.0 / tau))
        return SpectralHMM(transd_scaled, self.transU, self.pobs)

    def isclose(self, other):
        return np.allclose(self.transition_matrix, other.transition_matrix) and np.allclose(
            self.observation_probabilities, other.observation_probabilities) and \
               np.allclose(self.pobs, other.pobs)
