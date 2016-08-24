"""
This class is for classes of HMMs, including:
    * parameterized classes of HMMs with one float parameter.
    * metastable classes of HMMs with one float parameter.
"""
import numpy as np
from pyemma.msm.models.msm import MSM as _MM

from spectral_mm import SpectralMM


class MMClass:
    def ismember(self, x) -> bool:
        raise NotImplementedError("Please implement this method")


class MMClass1(MMClass):
    def eval(self, tau: float) -> MMClass:
        raise NotImplementedError("Please implement this method")


class MMM(MMClass1):
    def eval(self, tau: float) -> MMClass:
        raise NotImplementedError("Please implement this method")

    def constant(self) -> float:
        raise NotImplementedError("Please implement this method")


class MMMScaled(MMM):
    def __init__(self, smm: SpectralMM):
        assert smm.isdiagonal(), "smm is not diagonal"
        assert (np.diag(smm.transD) > 0).all(), "Some eigenvalues of smm are not positive"
        self.sMM = smm

    def ismember(self, x) -> bool:
        try:
            xeigenvalues = np.diag(x.transD)
            eigenvalues0 = np.diag(self.sMM.transD)
            tau = np.log(eigenvalues0[1]) / np.log(xeigenvalues[1])
            smmtest = self.sMM.scale(tau)
            if np.allclose(smmtest.transD, x.transD) and np.allclose(smmtest.transU, x.transU):
                return True
        except (AttributeError, TypeError):
            return False
        return False

    def eval(self, tau) -> SpectralMM:
        assert tau >= 1, "scaling factor is not greater or equal 1"
        # return a MM that is a scaling of sMM
        try:
            return self.sMM.scale(tau)  # type sMM
        except Exception:
            raise AssertionError('Input should be a number greater than or equal to 1')

    def constant(self):
        eigmin = np.min(np.absolute(np.real(np.diag(self.sMM.transD))))
        return -np.log(eigmin)
