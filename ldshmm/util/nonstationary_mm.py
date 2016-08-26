"""
This class is for non-stationary HMMs: mappings from integers within
a (dimensionless) temporal domain (timedomain), either [0, timeendpoint] or [0, 'infinity')
into the space of HMMs, where HMMs are defined conventionally with a lag of 1.
The sets of hidden and observable states are assumed to be constant and finite
throughout the temporal domain, and are identified by integer indices in
[0, nhidden) and [0, nobserved), resp.
"""

import numpy as np
from pyemma.msm.models.msm import MSM as _MM
from pyemma.util import types as _types
from ldshmm.util.spectral_mm import SpectralMM


class NonstationaryMM:
    def __init__(self, nstates: int, timeendpoint='infinity'):
        assert timeendpoint is 'infinity' or timeendpoint >= 0, \
            "The time domain endpoint should be a positive number of the string 'infinity'"
        self.timeendpoint = timeendpoint
        assert nstates > 0, "The number of states is not a positive integer"
        self.nstates = nstates

    def eval(self, time: int) -> _MM:
        assert time >= 0, "The evaluation time point is not a non-negative integer"
        if self.timeendpoint is not 'infinity':
            assert time <= self.timeendpoint, \
                "The evaluation time point is not less than or equal to the time domain endpoint."
        raise NotImplementedError("Please implement this method")

    def propagate(self, p0, k):
        r""" Propagates the initial distribution p0 k times

        Computes the product

        .. math::

            p_k = p_0^T P^k

        If the lag time of transition matrix :math:`P` is :math:`\tau`, this
        will provide the probability distribution at time :math:`k \tau`.

        Parameters
        ----------
        p0 : ndarray(n)
            Initial distribution. Vector of size of the active set.

        k : int
            Number of time steps

        Returns
        ----------
        pk : ndarray(n)
            Distribution after k steps. Vector of size of the active set.

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
        Generates a realization of the non-stationary Markov Model

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

        Returns
        -------
        dtraj : (N/dt, ) ndarray
            The state trajectory with length N/dt

        """

        dtraj = np.zeros(int(N/dt), dtype=int)

        dcurrent = self.eval(0).simulate(1, start, stop)
        dtraj[0] = dcurrent
        for i in range(0, N-1):
            dtraji = self.eval(i).simulate(2, dcurrent, stop)
            if dtraji.size == 1:
                return dtraj[:i/dt]
            elif (i+1) % dt == 0:
                dcurrent = dtraji[1]
                dtraj[int((i+1)/dt)] = dcurrent

        return dtraj

class NonstationaryMMClass:

    def ismember(self, x) -> bool:
        raise NotImplementedError("Please implement this method")


class ConvexCombinationNSMM(NonstationaryMM):

    def __init__(self, smm0, smm1, mu, timeendpoint='infinity'):
        super().__init__(smm0.nstates, timeendpoint)
        self.sMM0 = smm0
        self.sMM1 = smm1
        self.mu = mu

    def eval(self, time: int) -> SpectralMM:
        return self.sMM0.lincomb(self.sMM1, self.mu(time))

    def isclose(self, other, timepoints=None):
        if timepoints is None:
            if self.timeendpoint is not 'infinity':
                timepoints = range(0, self.timeendpoint)
            else:
                timepoints = range(0, 101)
        return self.sMM0.isclose(
            other.sMM0) and self.sMM1.isclose(other.sMM1) and \
               np.allclose(np.vectorize(self.mu)(timepoints), np.vectorize(self.mu)(timepoints))
