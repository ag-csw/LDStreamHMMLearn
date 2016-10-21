import numpy as np
from pyemma.msm.models.hmsm import HMSM as _HMM
from pyemma.util import types as _types


class NonstationaryHMM:
    """
    This class is for non-stationary HMMs: mappings from integers within
    a (dimensionless) temporal domain (timedomain), either [0, timeendpoint] or [0, 'infinity')
    into the space of HMMs, where HMMs are defined conventionally with a lag of 1.
    The sets of hidden and observable states are assumed to be constant and finite
    throughout the temporal domain, and are identified by integer indices in
    [0, nhidden) and [0, nobserved), resp.
    """

    def __init__(self, nhidden: int, nobserved: int, timeendpoint='infinity'):
        """

        :param nhidden: int - number of hidden states
        :param nobserved: int - number of observed states
        :param timeendpoint: (default='infinity') - time domain endpoint
        """

        assert timeendpoint is 'infinity' or timeendpoint >= 0, \
            "The time domain endpoint should be a positive number of the string 'infinity'"
        self.timeendpoint = timeendpoint
        assert nhidden > 0, "The number of hidden states is not a positive integer"
        self.nhidden = nhidden
        assert nobserved > 0, "The number of observed states is not a positive integer"
        self.nobserved = nobserved

    def eval(self, time: int) -> _HMM:
        """
        ToDo Document

        :param time: int - evaluation time point
        :return:
        """

        assert time >= 0, "The evaluation time point is not a non-negative integer"
        if self.timeendpoint is not 'infinity':
            assert time <= self.timeendpoint, \
                "The evaluation time point is not less than or equal to the time domain endpoint."
        raise NotImplementedError("Please implement this method")

    def propagate(self, p0, k):
        """
        propagates the initial distribution p0 k times

        Computes the product

        .. math::

            p_k = p_0^T P^k

        If the lag time of transition matrix :math:`P` is :math:`\tau`, this
        will provide the probability distribution at time :math:`k \tau`.

        :param p0: ndarray - initial distribution, vector of size of the active set
        :param k: int - number of time steps
        :return: ndarray - distribution after k steps, vector of size of the active set
        """

        p0 = _types.ensure_ndarray(p0, ndim=1, kind='numeric')
        assert _types.is_int(k) and k >= 0, 'k must be a non-negative integer'

        if k == 0 or k == 1:
            return self.eval(0).propagate(p0, k).real
        else:
            pprop = self.eval(0).propagate(p0, 1).real
            for i in range(1, k):
                pprop = self.eval(i).propagate(pprop, 1).real
            return pprop

    def simulate(self, N, start=None, stop=None, dt=1):
        """
        generates a realization of the Hidden Markov Model

        :param N: int  trajectory length in steps of the lag time
        :param start: int (default=None) - starting hidden state. If not given, will sample from the stationary
            distribution of the hidden transition matrix
        :param stop: int or int-array-like (default=None) - stopping hidden set. If given, the trajectory will be stopped before
            N steps once a hidden state of the stop set is reached
        :param dt: int - trajectory will be saved every dt time steps. Internally, the dt'th power of P is taken to ensure a more efficient simulation
        :return: ndarray, ndarray -  tuple of (hidden state trajectory with length N/dt, observable state discrete trajectory with length N/dt)
        """

        htraj = np.zeros(int(N/dt), dtype=int)
        otraj = np.zeros(int(N/dt), dtype=int)
        #print(htraj)

        hcurrent, ocurrent = self.eval(0).simulate(1, start, stop)
        htraj[0] = hcurrent
        #print("Initial Hidden State Array", htraj)
        #print("Step: ", dt)
        otraj[0] = ocurrent
        for i in range(0, N-1):
            htraji, otraji = self.eval(i).simulate(2, hcurrent, stop)
            #print("Hidden State SubArray", htraji)
            if htraji.size == 1:
                return htraj[:i/dt], otraj[:i/dt]
            elif (i+1) % dt == 0:
                hcurrent = htraji[1]
                htraj[int((i+1)/dt)] = hcurrent
                otraj[int((i+1)/dt)] = otraji[1]
                #print("Hidden State Array", htraj)

        return htraj, otraj

class NonstationaryHMMClass:
    """
    ToDo Document
    """

    def ismember(self, x) -> bool:
        """
        ToDo Document

        :param x:
        :return: bool
        """
        raise NotImplementedError("Please implement this method")


class ConvexCombinationNSHMM(NonstationaryHMM):
    """
    ToDo Document
    """

    def __init__(self, shmm0, shmm1, mu, timeendpoint='infinity'):
        """

        :param shmm0: SpectralHMM 1
        :param shmm1: SpectralHMM 2
        :param mu: function
        :param timeendpoint: (default='infinity') - time domain endpoint
        """

        super().__init__(shmm0.nstates, shmm0.nstates_obs, timeendpoint)
        self.sHMM0 = shmm0
        self.sHMM1 = shmm1
        self.mu = mu

    def eval(self, time: int) -> _HMM:
        return self.sHMM0.lincomb(self.sHMM1, self.mu(time))

    def isclose(self, other, timepoints=None):
        """
        returns whether two ConvexCombinationNSHMMs are close to each other

        :param other: ConvexCombinationNSHMM
        :param timepoints: : (default=None) - timepoints to consider
        :return: bool - True if the ConvexCombinationNSHMMs are close to each other, otherwise False
        """

        if timepoints is None:
            if self.timeendpoint is not 'infinity':
                timepoints = range(0, self.timeendpoint)
            else:
                timepoints = range(0, 101)
        return self.sHMM0.isclose(
            other.sHMM0) and self.sHMM1.isclose(other.sHMM1) and \
            np.allclose(np.vectorize(self.mu)(timepoints), np.vectorize(self.mu)(timepoints))
