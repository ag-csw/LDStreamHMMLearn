from ldshmm.util.nonstationary_hmm import ConvexCombinationNSHMM
from ldshmm.util.nonstationary_hmm import NonstationaryHMMClass


class QuasiHMM(NonstationaryHMMClass):
    """
    This class is for generic quasiHMMs. which are two-parameter families of mHMMs.
    """

    def eval(self, taumeta, tauquasi):
        """
        ToDo Document

        :param taumeta: int - scaling factor for the implied timescales of the snapshot MM
        :param tauquasi: int - drift timescale (non-stationary behaviour) relative to the largest implied timescale of the snapshot MM
        :return:
        """

        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # return an MHMM
        raise NotImplementedError("Please implement this method")

    def ismember(self, x) -> bool:
        """
        ToDo Document

        :param x:
        :return: bool
        """

        raise NotImplementedError("Please implement this method")


class ConvexCombinationQuasiHMM(QuasiHMM):
    """
    ToDo Document
    """

    def __init__(self, shmms, mu, timeendpoint='infinity'):
        """
        ToDo Document

        :param shmms: ndarray - SpectralHMMs
        :param mu: function
        :param timeendpoint: (default='infinity') - time domain endpoint
        """

        # the spectral HMM for mu = 0
        self.sHMM0 = shmms[0]
        # the spectral HMM for mu = 1
        self.sHMM1 = shmms[1]
        # the weight function for the convex combination
        self.mu = mu
        # the upper endpoint of the temporal domain interval
        # may be a postive integer or the string 'infinity'
        self.timeendpoint = timeendpoint

    def eval(self, taumeta, tauquasi) -> ConvexCombinationNSHMM:
        # return a non-stationary HMM
        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # scale the HMMs according to the parameter taumeta
        # with the effect that the implied timescales are increased by a factor of taumeta
        shmm0_scaled = self.sHMM0.scale(taumeta)  # type sHMM
        shmm1_scaled = self.sHMM1.scale(taumeta)  # type sHMM

        # scale the independent variable of the weight function
        # by the product of taumeta and tauquasi
        # with the effect that the timescale of the weight function, which represents drift in the model
        # is increased by a factor of taumeta
        taudrift = taumeta * tauquasi

        def mu_scaled(t):
            return self.mu(t / taudrift)  # type function

        # scale the temporal domain by the drift scaling
        if self.timeendpoint is not 'infinity':
            timeendpoint_scaled = self.timeendpoint * taudrift
        else:
            timeendpoint_scaled = 'infinity'
        return ConvexCombinationNSHMM(shmm0_scaled, shmm1_scaled, mu_scaled, timeendpoint_scaled)  # type nsHMM

    def ismember(self, x) -> bool:
        raise NotImplementedError("Please implement this method")
