"""
This class is for spectral HMMs. which are HMMs specified in terms of a particular Jordan decomposition of the
transmission matrix.
"""
import numpy as np
from pyemma.msm.models.hmsm import HMSM as _HMM
from msmtools.estimation import transition_matrix as _tm

class SpectralHMM(_HMM):
    def __init__(self, transd, transu, pobs, transv=None, trans=None):
        assert len(transd.shape) is 2, "transd is not a matrix"
        assert len(transu.shape) is 2, "transu is not a matrix"
        assert transd.shape[0] is transd.shape[1], "transd is not square"
        assert transu.shape[0] is transu.shape[1], "transu is not square"
        assert transu.shape[0] is transd.shape[0], "transd and transu do not have the same number of rows"

        self.transD = transd # the Jordan form of the transition matrix
        # FIXME another case would be to pass in the right eigenvector matrix transV and then calculate transU
        self.transU = transu # the left eigenvector matrix of the transition matrix
        self.pobs = pobs # the emission matrix


        assert not np.isclose(np.linalg.det(transu), 0.0), "transu is not invertible"
        if transv is None:
            self.transV = np.linalg.inv(self.transU)
        else:
            # if the right eigenvector matrix is known, it can be passed in
            self.transV = transv
        if trans is None:
            # calculate the transition matrix by definition:
            # the right eigenvector matrix times the Jordan form times the left eigenvector matrix
            self.trans = _tm(np.dot(np.dot(self.transV, self.transD), self.transU))
        else:
            # if the transition matrix is known, it can be passed in
            self.trans = _tm(trans)
        # apply the HMM constructor
        super(SpectralHMM, self).__init__(self.trans, self.pobs, self.transU[0], '1 step')

    def isdiagonal(self):
        return np.allclose(self.transD, np.diag(np.diag(self.transD)))

    def lincomb(self, other, mu):
        assert -1e-8 <= mu <= 1 + 1e-8, "weight is not between 0 and 1, inclusive"
        assert self.isdiagonal(), "self is not diagonal"
        assert other.isdiagonal(), "other is not diagonal"
        # FIXME check that both self and other have only positive eigenvalues less than or equal 1

        def lincc(x, y):
            return (1 - mu) * x + mu * y

        def logcc(x, y):
            return np.exp((1 - mu) * np.log(x) + mu * np.log(y))

        lincc = np.vectorize(lincc)
        logcc = np.vectorize(logcc)

        transd = np.diag(logcc(np.diag(self.transD), np.diag(other.transD)))
        transu = lincc(self.transU , other.transU)
        pobs = lincc(self.pobs,  other.pobs)
        return SpectralHMM(transd, transu, pobs)

    def scale(self, tau):
        assert tau > 0, "scaling factor is not positive"
        assert self.isdiagonal(), "self is not diagonal"

        # FIXME: would it be better to take the log?
        # Since we seem to use this a lot (see lincomb), how about a method that returns the log of the eigenvalues?

        transd_scaled = np.diag(np.power(np.diag(self.transD), 1.0 / tau))
        return SpectralHMM(transd_scaled, self.transU, self.pobs)

    def isclose(self, other):
        return np.allclose(self.transition_matrix, other.transition_matrix) and np.allclose(
            self.observation_probabilities, other.observation_probabilities)

    def simulate(self, N, start=None, stop=None, dt=1):
        """
        Generates a realization of the Hidden Markov Model

        Parameters
        ----------
        N : int
            trajectory length in steps of the lag time
        start : int, optional, default = None
            starting hidden state. If not given, will sample from the stationary
            distribution of the hidden transition matrix.
        stop : int or int-array-like, optional, default = None
            stopping hidden set. If given, the trajectory will be stopped before
            N steps once a hidden state of the stop set is reached
        dt : int
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.

        Returns
        -------
        htraj : (N/dt, ) ndarray
            The hidden state trajectory with length N/dt
        otraj : (N/dt, ) ndarray
            The observable state discrete trajectory with length N/dt

        """

        from scipy import stats
        import msmtools.generation as msmgen
        # generate output distributions
        output_distributions = [stats.rv_discrete(values=(np.arange(self.pobs.shape[1]), pobs_i)) for pobs_i in self.pobs]
        # sample hidden trajectory
        htraj = msmgen.generate_traj(self.transition_matrix, N, start=start, stop=stop, dt=dt)
        otraj = np.zeros(htraj.size, dtype=int)
        # for each time step, sample microstate
        for t, h in enumerate(htraj):
            otraj[t] = output_distributions[h].rvs()  # current cluster
        return htraj, otraj

    def is_scalable_tm(self):
        # For large scaling factors (tau), the scaling of the transition matrix approaches
        #
        #   I + (1/tau) ln(trans)
        #
        # This will be a  transition matrix for sufficiently large tau if
        #    1. all diagonal elements of ln(trans) are <= 0
        #    2. all off-diagonal elements ln(trans) are >= 0
        #
        # Therefore the matrix is called "scalable" if it satisfies these properties.
        #
        # The diagonal decomposition is used for a fast calculation of the natural log
        lntransd = np.diag(np.log(np.diag(self.transD)))
        delta = mdot(self.transV, lntransd, self.transU)
        # FIXME: This is not optimized, it does twice as many sign checks as necessary
        deltadiag = np.diag(delta)
        deltatril = np.tril(delta, -1)
        deltatriu = np.triu(delta, 1)
        if np.all(deltadiag <= 0) and np.all(deltatril >= 0) and np.all(deltatriu >= 0):
            return True
        else:

            return False