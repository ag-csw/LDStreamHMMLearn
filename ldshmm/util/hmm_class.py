"""
This class is for classes of HMMs, including:
    * parameterized classes of HMMs with one float parameter.
    * metastable classes of HMMs with one float parameter.
"""
import pyemma.msm as MSM
from pyemma.msm.models.hmsm import HMSM as _HMM
from spectral_hmm import SpectralHMM
import numpy as np

class HMMClass():
    def ismember(self, x ) -> bool:
        raise NotImplementedError("Please implement this method")

class HMMClass1(HMMClass):
    def eval(self, tau: float) -> _HMM:
        raise NotImplementedError("Please implement this method")

class mHMM(HMMClass1):
    def eval(self, tau: float) -> _HMM:
        raise NotImplementedError("Please implement this method")

class mHMMScaled(mHMM):
    def __init__(self, sHMM: SpectralHMM):
        assert sHMM.isdiagonal(), "sHMM is not diagonal"
        assert (np.diag(sHMM.transD) > 0).all(), "Some eigenvalues of sHMM are not positive"
        self.sHMM = sHMM

    def ismember(self, x ) -> bool:
        try:
            eigenvalues = np.diag(x.transD)
            eigenvalues0 = np.diag(self.sHMM.transD)
            tau = np.log(eigenvalues0[1])/np.log(eigenvalues[1])
            sHMMtest = self.sHMM.scale(tau)
            if np.allclose(sHMMtest.transD, x.transD) and np.allclose(sHMMtest.transU, x.transU) and np.allclose(sHMMtest.pobs, x.pobs):
                return True
        except (AttributeError, TypeError):
           return False
        return False

    def eval(self, tau) -> SpectralHMM:
        assert tau >= 1, "scaling factor is not greater or equal 1"
        # return a HMM that is a scaling of sHMM
        try:
            return self.sHMM.scale(tau) # type sHMM
        except (Exception):
            raise AssertionError('Input should be a number greater than or equal to 1')