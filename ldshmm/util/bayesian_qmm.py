import pyemma.msm as MSM
from pyemma.util.types import ensure_dtraj_list
from pyemma.msm.estimators.bayesian_msm import BayesianMSM as _BMSM
from pyemma.msm.estimators import MaximumLikelihoodMSM as _MSM
from pyemma.msm.models.msm import MSM as _MSM
from pyemma.msm.models.msm_sampled import SampledMSM as _SampledMSM

class BayesianQMM(_BMSM):
    """
    This class is for learning the model of a non-stationary HMM.
    """

    def __init__(self, lag=1, nsamples=100, nsteps=None,
                 conf=0.95,
                 show_progress=True):
        """
        ToDo Document show_progress probably a leftover from pyemmas BMSM
        Bayesian estimator for MSMs given discrete trajectory statistics

        :param lag: int (default=1) -  lagtime to estimate the HMSM at
        :param nsamples: int (default=100) - number of sampled transition matrices used
        :param nsteps: int (default=None) - number of Gibbs sampling steps for each transition matrix used.
            If None, nstep will be determined automatically
        :param conf: float (default=0.95) - Confidence interval. By default one-sigma (68.3%) is used. Use 95.4%
            for two sigma or 99.7% for three sigma.
        :param show_progress: bool (default=True) - Show progressbar
        """

        _BMSM.__init__(self, lag=lag)
        self.nsamples = nsamples
        self.nsteps = nsteps
        self.conf = conf
        self.show_progress = show_progress

    def estimate_MSM(self, dtrajs, nslide=1, window_size=10, nburnin=10):
        """
        ToDo document

        :param dtrajs: ndarray
        :param nslide: int (default=1) -
        :param window_size: int (default=10) -
        :param nburnin: int (default=10) -
        :return:
        """

        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)
        # conduct MLE estimation (superclass) of burnin data first
        _MSM._estimate(dtrajs[:, 0:nburnin])

        self.effective_count_matrix_posterior_list = []

        count_prior = ((self.window_size - self.nslide) / nburnin) * self.effective_count_matrix_

        #i=0
        # but this overwrites some previous values
        #
        #time = nburnin + (i+1)*self.nslide
        #_MSM._estimate(self, dtrajs[:, time-self.nslide:time])

        #count_data = self.self.effective_count_matrix_

        #count_posterior = count_prior + count_data
        #self.effective_count_matrix_posterior_list.append(( time, count_posterior))
        #count_prior = ((self.window_size - self.nslide) / self.window_size) * count_posterior

        #i=1
        #time = nburnin + (i+1)*self.nslide
        #_MSM._estimate(self, dtrajs[:, time-self.nslide:time])

        #count_data = self.effective_count_matrix_

        #count_posterior = count_prior + count_data
        #self.effective_count_matrix_posterior_list.append((time, count_posterior))
        #count_prior = ((self.window_size - self.nslide) / self.window_size) * count_posterior

        # imax is where
        # time = nburnin + (imax + 1)*self.nslide <= data_len
        #  (imax + 1)*self.nslide <= data_len - nburnin
        #   imax + 1 = int((data_len - nburnin)/self.nslide)
        data_len = dtrajs.shape[1]
        for i in range(0, int((data_len - nburnin)/self.nslide)):
            time = nburnin + (i+1) * self.nslide
            _MSM._estimate(dtrajs[:, (time - self.nslide):time])
            # FIXME: check to see if the estimate method truncates to observed values
            # If so, then the effective count matrix must be blown up to size nstates by nstates
            count_data = self.effective_count_matrix_
            # Does this need to be rounded?
            count_posterior = count_prior + count_data
            self.effective_count_matrix_posterior_list.append((time, count_posterior))
            count_prior = ((self.window_size - self.nslide) / self.window_size) * count_posterior


        # what other parameters should be set?
        #    - create a Bayesian sample every nbayes * nslide time interval
        # what other methods are needed?
        #    -
        return self

    def eval(self, time) -> _BMSM:
        """
        ToDo Document

        :param time:
        :return:
        """

        # initialize this array
        self.bmsm[time] = _BMSM(lag=self.lag, nsamples=self.nsamples, nsteps=None, reversible=False,
                 statdist_constraint=None, count_mode='effective', sparse=False,
                 connectivity='largest', dt_traj='1 step', conf=self.conf,
                 show_progress=self.show_progress)
        # calculate sample if it is not known
        # update model parameters in the bmsm as needed
        pass

    def eval_sample(self, time):
        """
        ToDo Document

        :param time:
        :return:
        """

        # transition matrix sampler
        from msmtools.estimation import tmatrix_sampler
        from math import sqrt
        self.nsteps = int(sqrt(self.nstates))  # heuristic for number of steps to decorrelate
        # use the same count matrix as the MLE. This is why we have effective as a default
        # FIXME: get the count matrix and transition matrix at "time"
        count_matrix = self.count_matrix_active
        tm = self.transition_matrix
        tsampler = tmatrix_sampler(count_matrix, reversible=self.reversible, T0=tm,
                                       nsteps=self.nsteps)

        self._progress_register(self.nsamples, description="Sampling MSMs", stage=0)

        if self.show_progress:
            def call_back():
                self._progress_update(1, stage=0)
        else:
            call_back = None

        sample_Ps, sample_mus = tsampler.sample(nsamples=self.nsamples,
                                                return_statdist=True,
                                                call_back=call_back)
        self._progress_force_finish(0)

        # construct sampled MSMs
        samples = []
        for i in range(self.nsamples):
            samples.append(_MSM(sample_Ps[i], pi=sample_mus[i], reversible=self.reversible, dt_model=self.dt_model))

        # update self model
        self.update_model_params(samples=samples)