"""
This class is for parameterized classes of HMMs with one float parameter.
"""
import pyemma.msm as MSM
from pyemma.msm.models.hmsm import HMSM as _HMM
import numpy as np



class HMMClass1():
    def eval(self, tau: float) -> _HMM:
        raise NotImplementedError("Please implement this method")

class mHMM(HMMClass1):
    def eval(self, tau: float) -> _HMM:
        raise NotImplementedError("Please implement this method")
