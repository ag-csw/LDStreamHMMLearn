import pyemma.msm as MSM
import numpy as np

from ldshmm.util.spectral_hmm import SpectralHMM
from ldshmm.util.hmm_class import mHMMScaled


class MSM_Test(unittest.TestCase):
    ########################################
    # Simulate Functions
    ########################################
    def test_simulate_MSM(self, transition_matrix, time_steps, initial_state):
        model = MSM.MSM(P=transition_matrix)
        sim = (model.simulate(N=3, start=initial_state))
        return sim

    def MSM_simulate_test_all(self, transition_matrix, time_step, initial_state):
        traj_MSM_num_traj = self.test_simulate_MSM(transition_matrix, time_step, None)
        print("MSM Trajectory Simulation with num_traj:\n ", traj_MSM_num_traj)
        traj_MSM_initial_state = self.test_simulate_MSM(transition_matrix, time_step, initial_state)
        print("MSM Trajectory Simulation with initial_state:\n", traj_MSM_initial_state)
        traj_MSM_both = self.test_simulate_MSM(transition_matrix, time_step, initial_state)

    def test_simulate_HMSM(self, transition_matrix, pobs, time_steps, initial_state):
        model = MSM.HMSM(P=transition_matrix, pobs=pobs)
        sim = (model.simulate(N=time_steps, start=initial_state))
        return sim
        print("MSM Trajectory Simulation with both num_traj and initial_state:\n", traj_MSM_both)

    def HMSM_simulate_test_all(self, transition_matrix, pobs, time_step, initial_state):
        traj_HMSM_initial_state = self.test_simulate_HMSM(transition_matrix, pobs, time_step, None)
        print("HMSM Trajectory Simulation without initial_state:\n", traj_HMSM_initial_state)
        traj_HMSM_initial_state = self.test_simulate_HMSM(transition_matrix, pobs, time_step, initial_state)
        print("HMSM Trajectory Simulation with initial_state:\n", traj_HMSM_initial_state)

    ########################################
    # HMM Model Estimation
    ########################################
    def create_HMSM_model_estimator(self, traj, nstates, lag, bayesian=False):
        if bayesian:
            estimator = MSM.BayesianHMSM(nstates=nstates, lag=lag, reversible=False, show_progress=True)
        else:
            estimator = MSM.MaximumLikelihoodHMSM(nstates=nstates, lag=lag, reversible=False)
        return estimator.estimate(traj)

    def HMSM_test_all(self, traj, nstates, lag):
        MLH_estimator = self.create_HMSM_model_estimator(traj, nstates, lag, False)
        print("Maximum Likelihood Estimator:\n", MLH_estimator)
        print("Transition Matrix:\n ", MLH_estimator.transition_matrix)
        print("Emission Matrix:\n ", MLH_estimator.observation_probabilities)
        bayesian_estimator = self.create_HMSM_model_estimator(traj, nstates, lag, True)
        print("Bayesian Estimator:\n", bayesian_estimator)
        print("Transition Matrix:\n ", bayesian_estimator.transition_matrix)
        print("Emission Matrix:\n ", bayesian_estimator.observation_probabilities)

    ########################################
    # Spectral HMM Model Creation
    ########################################
    def create_spectral_HMM(self, transD, transU, pobs):
        return SpectralHMM(transD, transU, pobs)

    def scale_spectral_HMM(self, sHMM, tau):
        return sHMM.scale(tau)

    def lincomb_spectral_HMM(self, sHMM0: SpectralHMM, sHMM1: SpectralHMM, mu: float) -> SpectralHMM:
        return sHMM0.lincomb(sHMM1, mu)

    ########################################
    # Metastable HMM Class Creation
    ########################################
    def create_scaled_HMM_class(self, sHMM):
        return mHMMScaled(sHMM)


    def sHMM_test_all(self, transD0, transU0, pobs0, tau, transD1, transU1, pobs1, mu):
        shmm0 = self.create_spectral_HMM(transD0, transU0, pobs0)
        print("\nSpectral HMM:\n")
        print("Transition Matrix:\n", shmm0.transition_matrix)
        print("Emission Matrix:\n", shmm0.observation_probabilities)
        shmm1 = self.create_spectral_HMM(transD1, transU1, pobs1)
        print("\nSpectral HMM:\n")
        print("Transition Matrix:\n", shmm1.transition_matrix)
        print("Emission Matrix:\n", shmm1.observation_probabilities)
        shmm0_scaled = self.scale_spectral_HMM(shmm0, tau)
        print("\nScaled Spectral HMM:\n")
        print("Transition Matrix:\n", shmm0_scaled.transition_matrix)
        print("Emission Matrix:\n", shmm0_scaled.observation_probabilities)
        shmm_lc = self.lincomb_spectral_HMM(shmm0, shmm1, mu)
        print("\nLinear Combination Spectral HMM:\n")
        print("Transition Matrix:\n", shmm_lc.transition_matrix)
        print("Emission Matrix:\n", shmm_lc.observation_probabilities)

    def mHMM_test_all(self, transD0, transU0, pobs0, tau):
        shmm0 = self.create_spectral_HMM(transD0, transU0, pobs0)
        shmm0_scaled = self.scale_spectral_HMM(shmm0, tau)
        mhmm = self.create_scaled_HMM_class(shmm0)
        shmm0_scaled2 = mhmm.eval(tau)
        print("\nCheck of Scaled Class Membership: ", mhmm.ismember(shmm0_scaled))

        ########################################
        # Functionality tests
        ########################################
    def All_test_all(self):
        transition_matrix = np.array([[0.9, 0.05, 0.05],
                                      [0.1, 0.8, 0.1],
                                      [0.1, 0.1, 0.8]])
        pobs = np.array(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1]
            ])
        initial_state = 0

        test = MSM_Test()

        test.MSM_simulate_test_all(transition_matrix, 3, initial_state)
        print("\n")
        test.HMSM_simulate_test_all(transition_matrix, pobs, 3, initial_state)

        print("\n\n")
        traj, obs = test.test_simulate_HMSM(transition_matrix, pobs, 1000, None)
        print(traj)
        test.HMSM_test_all(obs, 3, 1)

        transition_matrix_jordan0 = np.array([[1.0, 0, 0],
                                              [0, 0.2, 0],
                                              [0, 0, 0.1]])

        transition_matrix_basis0 = np.array([[0.8, 0.1, 0.1],
                                             [-1, 1, 0],
                                             [1, 1, -2]])

        pobs0 = pobs

        tau = 10
        transition_matrix_jordan1 = np.array([[1.0, 0, 0],
                                              [0, 0.9, 0],
                                              [0, 0, 0.8]])

        transition_matrix_basis1 = np.fliplr(transition_matrix_basis0)
        print("Transition Matrix Basis:\n", transition_matrix_basis1)
        pobs1 = np.flipud(pobs)
        mu = 0.3
        test.sHMM_test_all(transition_matrix_jordan0, transition_matrix_basis0, pobs0, tau, transition_matrix_jordan1,
                       transition_matrix_basis1, pobs1, mu)


        test.mHMM_test_all(transition_matrix_jordan0, transition_matrix_basis0, pobs0, tau)