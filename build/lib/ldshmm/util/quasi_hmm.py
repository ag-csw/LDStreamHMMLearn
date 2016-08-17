"""
This class is for generic quasiHMMs. which are two-parameter families of mHMMs.
"""


class QuasiHMM(object):

    def mHMM(self, taumeta, tauquasi):
        assert taumeta >= 1, "taumeta is not greater or equal 1"
        assert tauquasi >= 1, "tauquasi is not greater or equal 1"
        # return an mHMM
        raise NotImplementedError("Please implement this method")
