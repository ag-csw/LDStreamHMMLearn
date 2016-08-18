"""
This class is for non-stationary HMMs: mappings from integers within
a (dimensionless) temporal domain (timedomain), either [0, timeendpoint] or [0, 'infinity')
into the space of HMMs, where HMMs are defined conventionally with a lag of 1.
The sets of hidden and observable states are assumed to be constant and finite
throughout the temporal domain, and are identified by integer indices in
[0, nhidden) and [0, nobserved), resp.
"""

import pyemma.msm as MSM
from pyemma.msm.models.hmsm import HMSM as _HMM
import numpy as np

class NonstationaryHMM():
    def __init__(self, timeendpoint, nhidden: int, nobserved: int):
        assert timeendpoint is 'infinity' or timeendpoint >= 0, "The time domain endpoint should be a positive number of the string 'infinity'"
        self.timeendpoint = timeendpoint
        assert nhidden > 0, "The number of hidden states is not a positive integer"
        self.nhidden = nhidden
        assert nobserved > 0, "The number of observed states is not a positive integer"
        self.nobserved = nobserved

    def eval(self, time:int) -> _HMM:
        assert time >= 0, "The evaluation time point is not a non-negative integer"
        if self.timeendpoint is not 'infinity':
            assert time <= self.timeendpoint, "The evaluation time point is not less than or equal to the time domain endpoint."
        raise NotImplementedError("Please implement this method")

class NonstationaryHMMClass():
    def ismember(self, x ) -> bool:
        raise NotImplementedError("Please implement this method")