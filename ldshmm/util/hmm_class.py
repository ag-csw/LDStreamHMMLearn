"""
This class is for classes of HMMs, including:
    * parameterized classes of HMMs with one float parameter.
    * metastable classes of HMMs with one float parameter.
"""
import numpy as np
from pyemma.msm.models.hmsm import HMSM as _HMM

from ldshmm.util.spectral_hmm import SpectralHMM


class HMMClass:
    def ismember(self, x) -> bool:
        """
        ToDo Document

        :param x:
        :return: bool
        """

        raise NotImplementedError("Please implement this method")


class HMMClass1(HMMClass):
    def eval(self, tau: float) -> _HMM:
        """
        ToDo Document

        :param tau: float
        :return: a HMM
        """

        raise NotImplementedError("Please implement this method")


class MHMM(HMMClass1):
    def eval(self, tau: float) -> _HMM:
        raise NotImplementedError("Please implement this method")


class MHMMScaled(MHMM):
    def __init__(self, shmm: SpectralHMM):
        """

        :param shmm: SpectralHMM
        """

        assert shmm.isdiagonal(), "shmm is not diagonal"
        assert (np.diag(shmm.transD) > 0).all(), "Some eigenvalues of shmm are not positive"
        self.sHMM = shmm

    def ismember(self, x) -> bool:
        try:
            eigenvalues = np.diag(x.transD)
            eigenvalues0 = np.diag(self.sHMM.transD)
            tau = np.log(eigenvalues0[1]) / np.log(eigenvalues[1])
            shmmtest = self.sHMM.scale(tau)
            if np.allclose(shmmtest.transD, x.transD) and np.allclose(shmmtest.transU, x.transU) and np.allclose(
                    shmmtest.pobs, x.pobs):
                return True
        except (AttributeError, TypeError):
            return False
        return False

    def eval(self, tau) -> SpectralHMM:
        assert tau >= 1, "scaling factor is not greater or equal 1"
        # return a HMM that is a scaling of sHMM
        try:
            return self.sHMM.scale(tau)  # type sHMM
        except Exception:
            raise AssertionError('Input should be a number greater than or equal to 1')
