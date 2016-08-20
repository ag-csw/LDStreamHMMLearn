"""
This class is for learning the model of a non-stationary HMM.
"""
import pyemma.msm as MSM


class BayesianQHMM():
    def __init__(self,
                 nstates=4,
                 lag=1,
                 stride='effective',
                 cluster_prior=None,
                 nsamples=100,
                 init_hmsm=None,
                 reversible=False,
                 stationary=False,
                 connectivity='largest',
                 mincount_connectivity = '1/n',
                 separate = None,
                 observe_nonempty = True,
                 dt_traj = '1 step',
                 conf = 0.95,
                 store_hidden = False,
                 show_progress = True,
                 nwindow = 100,
                 step = 10):
        self.nstates = nstates
        self.lag = lag
        self.stride = stride
        self.cluster_prior = cluster_prior
        self.nsamples = nsamples
        self.init_hmsm = init_hmsm
        self.stationary = stationary
        self.reversible = reversible
        self.connectivity = connectivity
        self.mincount_connectivity = mincount_connectivity
        self.separate = separate
        self.observe_nonempty = observe_nonempty
        self.dt_traj = dt_traj
        self.conf = conf
        self.store_hidden = store_hidden
        self.show_progress = show_progress
        self.nwindow = nwindow
        self.step = step

    def estimate_HMSM(self, dtrajs):
        self.dtrajs = dtrajs
        ntime_steps = self.data.shape[1]
        nlearn = (self.ntime_steps - self.nwindow) / self.step
        tms = []
        ems = []
        for j in range(nlearn):
            hmm = MSM.BayesianHMSM(self.nstates, self.lag, self.stride, None, None, self.nsamples,
                                   self.init_hmsm, self.reversible, self.stationary, self.connectivity,
                                   self.mincount_connectivity, self.separate, self.observe_nonempty, self.dt_traj,
                                   self.conf, self.store_hidden, self.show_progress)
            tm, em = hmm.estimate(self.dtrajs[:, j * self.step: j * self.step + self.nwindow - 1])
            tms.append(tm)
            ems.append(em)
        return tms, ems

    def estimate_MSM(self, dtrajs, nslide=10):
        from pyemma.util.types import ensure_dtraj_list
        from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MLMSM
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)
        # conduct MLE estimation (superclass) first
        _MLMSM._estimate(self, dtrajs[:, 0:nslide])