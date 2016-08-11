import pyemma.msm as MSM
import numpy as np


class MSM_Test():
    ########################################
    # Simulate Functions
    ########################################
    def test_simulate_MSM (self, transition_matrix, time_steps, initial_state):
        model = MSM.MSM(P=transition_matrix)
        sim = (model.simulate(N=3, start=initial_state ))
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
    # HMM Model Creation
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
# Functionality tests
########################################
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
