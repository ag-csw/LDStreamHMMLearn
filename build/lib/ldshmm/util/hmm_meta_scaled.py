"""
This class is for generic quasiHMMs. which are two-parameter families of mHMMs.
"""
import numpy as np

from hmm_class1 import mHMM as _mHMM
from spectral_hmm import SpectralHMM

class mHMMScaled(_mHMM):
    def __init__(self, sHMM: SpectralHMM):
        assert sHMM.isdiagonal(), "sHMM is not diagonal"
        assert (np.diag(sHMM.transD) > 0).all(), "Some eigenvalues of sHMM are not positive"
        self.sHMM = sHMM

    def eval(self, tau: float) -> SpectralHMM:
        assert tau >= 1, "scaling factor is not greater or equal 1"
        # return a HMM that is a scaling of sHMM
        return self.sHMM.scale(tau) # type sHMM
