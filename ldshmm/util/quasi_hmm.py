"""
This class is for generic quasiHMMs. which are two-parameter families of mHMMs.
"""
from nonstationary_hmm import ConvexCombinationNSHMM
from nonstationary_hmm import NonstationaryHMMClass


class QuasiHMM(NonstationaryHMMClass):
    def eval(self, taumeta, tauquasi):
        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # return an MHMM
        raise NotImplementedError("Please implement this method")

    def ismember(self, x) -> bool:
        raise NotImplementedError("Please implement this method")


class ConvexCombinationQuasiHMM(QuasiHMM):
    def __init__(self, shmm0, shmm1, mu, timeendpoint):
        self.sHMM0 = shmm0
        self.sHMM1 = shmm1
        self.mu = mu
        self.timeendpoint = timeendpoint

    def eval(self, taumeta, tauquasi) -> ConvexCombinationNSHMM:
        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # return a non-stationary HMM
        shmm0_scaled = self.sHMM0.scale(taumeta)  # type sHMM
        shmm1_scaled = self.sHMM1.scale(taumeta)  # type sHMM

        def mu_scaled(t):
            return self.mu(t / (taumeta * tauquasi))  # type function

        if self.timeendpoint is not 'infinity':
            timeendpoint_scaled = self.timeendpoint / (taumeta * tauquasi)
        else:
            timeendpoint_scaled = 'infinity'
        return ConvexCombinationNSHMM(shmm0_scaled, shmm1_scaled, mu_scaled, timeendpoint_scaled)  # type nsHMM

    def ismember(self, x) -> bool:
        raise NotImplementedError("Please implement this method")
