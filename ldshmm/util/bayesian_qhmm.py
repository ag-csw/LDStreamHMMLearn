
import pyemma.msm as MSM


class BayesianQHMM():
    """
    This class is for learning the model of a non-stationary HMM.
    """

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
                 window_size = 100,
                 step = 10):
        """
        ToDo Document show_progress probably leftover from pyemmas BayesianHMSM

        :param nstates: int (default=4) - number of states
        :param lag: int (default=1) - lag time
        :param stride: (default='effective')
        :param cluster_prior:  (default=None)
        :param nsamples: int (default=100) - number of samples
        :param init_hmsm: (default=None)
        :param reversible: (default=False)
        :param stationary: (default=False)
        :param connectivity: (default='largest')
        :param mincount_connectivity: (default='1/n')
        :param separate: (default=None)
        :param observe_nonempty: (default=True)
        :param dt_traj: (default='1 step')
        :param conf: float (default=0.95)
        :param store_hidden: (default=False)
        :param show_progress: (default=True)
        :param window_size: int (default=100)
        :param step: int (default=10)
        """

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
        self.window_size = window_size
        self.step = step

    def estimate_HMSM(self, dtrajs):
        """
        ToDo Document

        :param dtrajs: ndarray
        :return:
        """

        self.dtrajs = dtrajs
        ntime_steps = self.data.shape[1]
        nlearn = (self.ntime_steps - self.window_size) / self.step
        tms = []
        ems = []
        for j in range(nlearn):
            hmm = MSM.BayesianHMSM(self.nstates, self.lag, self.stride, None, None, self.nsamples,
                                   self.init_hmsm, self.reversible, self.stationary, self.connectivity,
                                   self.mincount_connectivity, self.separate, self.observe_nonempty, self.dt_traj,
                                   self.conf, self.store_hidden, self.show_progress)
            tm, em = hmm.estimate(self.dtrajs[:, j * self.step: j * self.step + self.window_size - 1])
            tms.append(tm)
            ems.append(em)
        return tms, ems

    def estimate_MSM(self, dtrajs, nslide=10):
        """
        ToDo Document

        :param dtrajs: ndarray
        :param nslide: int (default=10)
        :return:
        """

        from pyemma.util.types import ensure_dtraj_list
        from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MLMSM
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)
        # conduct MLE estimation (superclass) first
        _MLMSM._estimate(self, dtrajs[:, 0:nslide])