"""
This class is for spectral HMMs. which are HMMs specified in terms of a particular Jordan decomposition of the transmission matrix.
"""
import pyemma.msm as MSM
from pyemma.msm.models.hmsm import HMSM as _HMM
import numpy as np



class SpectralHMM(_HMM):
    def __init__(self, transD, transU, pobs):
        assert len( np.shape(transD) ) is 2, "transD is not a matrix"
        assert len( np.shape(transU) ) is 2, "transU is not a matrix"
        assert np.shape(transD)[0] is np.shape(transD)[1], "transD is not square"
        assert np.shape(transU)[0] is np.shape(transU)[1], "transU is not square"
        assert np.shape(transU)[0] is np.shape(transD)[0], "transD and transU do not have the same number of rows"
        self.transD = transD
        self.transU = transU
        assert not np.allclose(a=np.linalg.det(transU), b=0.0), "transU is not invertible"
        P = np.dot(np.dot(np.linalg.inv(transU), transD), transU)
        super(SpectralHMM, self).__init__(P, pobs, transU[0], '1 step')

    def isdiagonal(self):
        return np.allclose(self.transD, np.diag(np.diag(self.transD)))

    def lincomb(self, other, mu):
        assert -1e-8 <= mu and mu <= 1+1e-8, "weight is not between 0 and 1, inclusive"
        assert self.isdiagonal(), "self is not diagonal"
        assert other.isdiagonal(), "other is not diagonal"
        transD = (1.0- mu) * self.transD + mu * other.transD
        transU = (1.0- mu) * self.transU + mu * other.transU
        pobs = (1.0- mu) * self.pobs + mu * other.pobs
        return SpectralHMM(transD, transU, pobs)

    def scale(self, tau):
        assert tau > 0, "scaling factor is not positive"
        assert self.isdiagonal(), "self is not diagonal"
        transD_scaled = np.diag(np.power(np.diag(self.transD), 1.0/tau ))
        return SpectralHMM(transD_scaled, self.transU, self.pobs)

    def isclose(self, other):
        return np.allclose(self.transition_matrix, other.transition_matrix) and np.allclose(self.observation_probabilities, other.observation_probabilities) and \
               np.allclose(self.pobs, other.pobs)