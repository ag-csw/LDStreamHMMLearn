import numpy as np
from pyemma.msm.models.msm import MSM as _MM

from ldshmm.util.spectral_mm import SpectralMM


class MMClass:
    """
    This class is for classes of HMMs, including:
    * parameterized classes of HMMs with one float parameter.
    * metastable classes of HMMs with one float parameter.
    """
    def ismember(self, x) -> bool:
        """
        ToDo Document

        :param x:
        :return: bool
        """

        raise NotImplementedError("Please implement this method")


class MMClass1(MMClass):
    """
    ToDo Document
    """
    def eval(self, tau: float) -> MMClass:
        """
        ToDo Document

        :param tau: float
        :return: a MM
        """

        raise NotImplementedError("Please implement this method")


class MMM(MMClass1):
    """
    ToDo Document
    """
    def eval(self, tau: float) -> MMClass:
        raise NotImplementedError("Please implement this method")

    def constant(self) -> float:
        """
        ToDo Document

        :return: float
        """

        raise NotImplementedError("Please implement this method")


class MMMScaled(MMM):
    def __init__(self, smm: SpectralMM):
        """

        :param smm: SpectralMM
        """

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

    def lincomb(self, other, mu):
        """
        Constructs a matrix M that is a combination of self and other in the following sense:
        1) The eigenvalues (diagonal of transd) of M are the weighted geometric mean (https://en.wikipedia.org/wiki/Weighted_geometric_mean) of the eigenvalues of self and other
        2) the left eigenvectors (rows of transu) of M are the weighted average (linear combination) of the left eigenvectors of self and other
        In each case the weights are (1-mu) and mu. Cases:
          - When mu = 0, self is returned.
          - When mu = 1, other is returned.
          - When 0 < mu < 1, M is in some sense "between" self and other.
        The conditions above are not sufficient to define a unique matrix M.
        This implementation requires the matrices self and other to be diagonalizable and
        provided as a Jordan decomposition.
        The weight mu is required to be between 0 and 1, inclusive.
        :param other: MMMScaled
        :param mu:
        :return: MMMScaled based on transd and transu

        """

        assert -1e-8 <= mu <= 1 + 1e-8, "weight is not between 0 and 1, inclusive"
        assert self.sMM.isdiagonal(), "self is not diagonal"
        assert other.sMM.isdiagonal(), "other is not diagonal"

        # FIXME check that both self and other have only positive eigenvalues less than or equal 1

        def lincc(x, y):
            return (1 - mu) * x + mu * y

        def logcc(x, y):
            return np.exp((1 - mu) * np.log(x) + mu * np.log(y))

        lincc = np.vectorize(lincc)
        logcc = np.vectorize(logcc)

        transd = np.diag(logcc(np.diag(self.sMM.transD), np.diag(other.sMM.transD)))
        transu = lincc(self.sMM.transU, other.sMM.transU)

        return MMMScaled(smm=SpectralMM(transd, transu))