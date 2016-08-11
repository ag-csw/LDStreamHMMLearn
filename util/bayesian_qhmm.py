"""
This class is for learning the parameters of a quasi-stationary HMM.
"""
import pyemma.msm as MSM
import numpy as np


class BayesianQHMM():
    def __init__(self, nstates, lag, stride, cluster_prior, nsamples, init_hmsm, reversible,stationary,  connectivity,
                 mincount_connectivity, separate, observe_nonempty, dt_traj, conf, store_hidden, show_progress, data,
                 nwindow, step):
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
        self.data = data
        self.nwindow = nwindow
        self.step = step

    def estimate(self):
        ntime_steps = self.data.shape[1]
        nlearn = (self.ntime_steps - self.nwindow)/self.step
        tms = []
        ems = []
        for j in range(nlearn):
            hmm = MSM.BayesianHMSM(self.nstates, self.lag, self.stride, None, None, self.nsamples,
                                   self.init_hmsm,  self.reversible, self.stationary,self.connectivity,
                                   self.mincount_connectivity, self.separate, self.observe_nonempty, self.dt_traj,
                                   self.conf, self.store_hidden, self.show_progress)
            tm, em = hmm.estimate(self.data[:, j*self.step: j*self.step + self.nwindow - 1])
            tms.append(tm)
            ems.append(em)
        return tms, ems