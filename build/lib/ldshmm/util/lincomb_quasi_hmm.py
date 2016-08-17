"""
This class is for generic quasiHMMs. which are two-parameter families of mHMMs.
"""
from quasi_hmm import QuasiHMM as _QHMM

class LinearCombinationQuasiHMM(_QHMM):
    def __init__(self, sHMM0, sHMM1, mu):
        self.sHMM0 = sHMM0
        self.sHMM1 = sHMM1
        self.mu = mu

    def create_nsHMM(self, taumeta, tauquasi):
        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # return a non-stationary HMM
        sHMM0_scaled = self.sHMM0.scale(taumeta) # type sHMM
        sHMM1_scaled = self.sHMM1.scale(taumeta) # type sHMM
        def mu_scaled (t): return self.mu( t/tauquasi ) # type function
        return sHMM0_scaled.create_lcnsHMM(sHMM1_scaled, mu_scaled) # type nsHMM
