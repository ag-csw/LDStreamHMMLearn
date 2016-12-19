from ldshmm.util.nonstationary_mm import ConvexCombinationNSMM
from ldshmm.util.nonstationary_mm import NonstationaryMMClass


class QuasiMM(NonstationaryMMClass):
    """
    This class is for generic quasiMMs. which is mapping from two
     float parameters into NSMM classes.
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
        # return an MMM
        raise NotImplementedError("Please implement this method")

    def ismember(self, x) -> bool:
        """
        return True if x is a member of self

        :param x:
        :return: bool
        """

        raise NotImplementedError("Please implement this method")


class ConvexCombinationQuasiMM(QuasiMM):
    """
    quasistationary family of ConvexCombinationMMs

    ToDo: get definition of quasistationary from paper
    """

    def __init__(self, mmms, mu, timeendpoint='infinity'):
        """
        ToDo Document

        :param mmms:
        :param mu: function
        :param timeendpoint: (default='infinity') - time domain endpoint
        """

        # the metastable MM for mu = 0
        self.mMM0 = mmms[0]
        # the metastable MM for mu = 1
        self.mMM1 = mmms[1]
        # the weight function for the convex combination
        self.mu = mu
        # the upper endpoint of the temporal domain interval
        # may be a postive integer or the string 'infinity'
        self.timeendpoint = timeendpoint

    def eval(self, taumeta, tauquasi) -> ConvexCombinationNSMM:
        # return a NSMM class corresponding to the specified
        # two parameter values
        # FIXME: there is a datatype mismatch since
        # ConvexCombinationNSMM is not explicitly a NonstationaryMMClass
        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # scale the MMs according to the parameter taumeta
        # with the effect that the implied timescales are increased by a factor of taumeta
        smm0_scaled = self.mMM0.eval(taumeta)  # type sMM
        smm1_scaled = self.mMM1.eval(taumeta)  # type sMM

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
        return ConvexCombinationNSMM(smm0_scaled, smm1_scaled, mu_scaled, timeendpoint_scaled)  # type nsMM

    def ismember(self, x) -> bool:
        raise NotImplementedError("Please implement this method")
