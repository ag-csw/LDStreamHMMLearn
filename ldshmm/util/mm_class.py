import numpy as np
from pyemma.msm.models.msm import MSM as _MM

from ldshmm.util.spectral_mm import SpectralMM


class MMClass:
    """
    This class is for "classes" of MMs (Markov Models), including:
    * parameterized classes of MMs with one float parameter (see MMClass1).
    * metastable classes of MMs with one float parameter.
    """
    def ismember(self, x) -> bool:
        """
        Returns true if x is a member of the class

        :param x:
        :return: bool
        """

        raise NotImplementedError("Please implement this method")


class MMClass1(MMClass):
    """
    MMClass which is a mapping of a single positive parameter
    $\timescale{\meta}$ into MM classes
    """
    def eval(self, tau: float) -> MMClass:
        """
        extract the class member MMClass corresponding to the
        paramater value tau.

        :param tau: float
        :return: a MM
        """

        raise NotImplementedError("Please implement this method")


class MMM(MMClass1):
    """
    MMClass1 that is a metastable class of MM.
    That is,
    \begin{defn}
\label{defn-metastable-mm}
A metastable MM class (mMM) is a parameterized class of (stationary) MMs
in the form of a mapping of a single positive parameter
$\timescale{\meta}$ into MM classes where
\begin{itemize}
    \item the transition matrix of any member MM in this class is
    diagonalizable and has only positive real
    eigenvalues $\eigenvalue_k$,
    $ 0\le k < \nummeta$, only one of which ($\eigenvalue_0$)
    is equal to $1$,
    \item there is some  $\const{}>0$ such that for any
    $\timescale{\meta}$, all eigenvalues $\eigenvalue_k$ of a
    member MM's transition matrix are greater than or equal
    $e^{-\frac{\const{}}{\timescale{\meta}} }$.
    \item Whenever $\timescale{\meta}\ge 1$, the corresponding MM class
    is non-empty.
\end{itemize}
\end{defn}
The first condition guarantees that each member MM has a
unique stationary state $\stat$,
while the second
condition guarantees that for (asymptotically) large values of
$\timescale{\meta}$, the state (in trajectories from member MMs) changes
slowly,\footnote{If an
arbitrary probability distribution of states has small incremental
differences when it is evolved according to an MM, we say that the
state of the MM changes slowly.}
relative to the lag time of the
MM.\footnote{For simplicity, we do not consider in this analysis the case of
transition matrices that are defective or have complex spectra,
although there are also cases of such MMs where the states change slowly.}

The states of an mMM are called metastable states of the mMM,
because given the state at some time, the next state with
maximum likelihood is always the same state,
for sufficiently large $\timescale{\meta}$.
    """
    def eval(self, tau: float) -> MMClass:
        raise NotImplementedError("Please implement this method")

    def constant(self) -> float:
        """
        return a constant value such that the second requirement
        in the definition of mMM is satisfied for this constant.

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
        # FIXME - there is a datatype mismatch because eval is supposed to
        # return an MMClass (according to the supertype) but it is returning
        # a SpectralMM - this could be viewed as a singleton MM class,
        # but it does not inherit this datatype.
        try:
            return self.sMM.scale(tau)  # type sMM
        except Exception:
            raise AssertionError('Input should be a number greater than or equal to 1')

    def constant(self):
        """ToDo Unit Test
        Verify the second requirement of mMM for this constant.
        """
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