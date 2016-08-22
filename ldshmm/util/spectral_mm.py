"""
This class is for spectral HMMs. which are HMMs specified in terms of a particular Jordan decomposition of the
transmission matrix.
"""
import numpy as np
from pyemma.msm.models.msm import MSM as _MM


class SpectralMM(_MM):
    def __init__(self, transd, transu, transv=None, trans=None):
        assert len(transd.shape) is 2, "transd is not a matrix"
        assert len(transu.shape) is 2, "transu is not a matrix"
        assert transd.shape[0] is transd.shape[1], "transd is not square"
        assert transu.shape[0] is transu.shape[1], "transu is not square"
        assert transu.shape[0] is transd.shape[0], "transd and transu do not have the same number of rows"

        self.transD = transd # the Jordan form of the transition matrix
        # FIXME another case would be to pass in the right eigenvector matrix transV and then calculate transU
        self.transU = transu # the left eigenvector matrix of the transition matrix

        assert not np.linalg.det(transu) == 0.0, "transu is not invertible"
        if transv is None:
            self.transV = np.linalg.inv(self.transU)
        else:
            # if the right eigenvector matrix is known, it can be passed in
            self.transV = transv
        if trans is None:
            # calculate the transition matrix by definition:
            # the right eigenvector matrix times the Jordan form times the left eigenvector matrix
            self.trans = np.dot(np.dot(self.transV, self.transD), self.transU)
        else:
            # if the transition matrix is known, it can be passed in
            self.trans = trans
        # apply the MM constructor
        super(SpectralMM, self).__init__(self.trans)
        # super(SpectralMM, self).__init__(self.trans, self.transU[0], '1 step')

    def isdiagonal(self):
        return np.allclose(self.transD, np.diag(np.diag(self.transD)))

    def lincomb(self, other, mu):
        assert -1e-8 <= mu <= 1 + 1e-8, "weight is not between 0 and 1, inclusive"
        assert self.isdiagonal(), "self is not diagonal"
        assert other.isdiagonal(), "other is not diagonal"

        transd = (1.0 - mu) * self.transD + mu * other.transD
        transu = (1.0 - mu) * self.transU + mu * other.transU
        return SpectralMM(transd, transu)

    def scale(self, tau):
        assert tau > 0, "scaling factor is not positive"
        assert self.isdiagonal(), "self is not diagonal"

        transd_scaled = np.diag(np.power(np.diag(self.transD), 1.0 / tau))
        return SpectralMM(transd_scaled, self.transU)

    def isclose(self, other):
        return np.allclose(self.transition_matrix, other.transition_matrix)
