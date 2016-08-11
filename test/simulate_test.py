import pyemma.msm as MSM
import numpy as np


class MSM_Test():
    ########################################
    # Simulate Functions
    ########################################
    def test_simulate_MSM (self, transition_matrix, time_steps, num_traj, initial_states):
        model = MSM.MSM(P=transition_matrix)
        sim = (model.simulate(time_steps=3, num_traj=3, initial_states=initial_states ))
        return sim

    def MSM_simulate_test_all(self, transition_matrix, time_step, num_traj, initial_states):
        traj_MSM_num_traj = self.test_simulate_MSM(transition_matrix, time_step, num_traj, None)
        print("MSM Trajectory Simulation with num_traj:\n ", traj_MSM_num_traj)
        traj_MSM_initial_states = self.test_simulate_MSM(transition_matrix, time_step, None, initial_states)
        print("MSM Trajectory Simulation with initial_states:\n", traj_MSM_initial_states)
        traj_MSM_both = self.test_simulate_MSM(transition_matrix, time_step, num_traj, initial_states)

    def test_simulate_HMSM(self, transition_matrix, pobs, time_steps, num_traj, initial_states):
        model = MSM.HMSM(P=transition_matrix, pobs=pobs)
        sim = (model.simulate(time_steps=time_steps, num_traj=num_traj, initial_states=initial_states))
        return sim
        print("MSM Trajectory Simulation with both num_traj and initial_states:\n", traj_MSM_both)

    def HMSM_simulate_test_all(self, transition_matrix, pobs, time_step, num_traj, initial_states):
        traj_HMSM_num_traj = self.test_simulate_HMSM(transition_matrix, pobs, time_step, num_traj, None)
        print("HMSM Trajectory Simulation with num_traj:\n ", traj_HMSM_num_traj)
        traj_HMSM_initial_states = self.test_simulate_HMSM(transition_matrix, pobs, time_step, None, initial_states)
        print("HMSM Trajectory Simulation with initial_states:\n", traj_HMSM_initial_states)
        traj_HMSM_both = self.test_simulate_HMSM(transition_matrix, pobs, time_step, num_traj, initial_states)
        print("HMSM Trajectory Simulation with both num_traj and initial_states:\n", traj_HMSM_both)

    ########################################
    # HMM Model Creation
    ########################################
    def create_HMSM_model_estimator(self, traj, nstates, lag, bayesian=False):
        if bayesian:
            estimator = MSM.BayesianHMSM(nstates=nstates, lag=lag, show_progress=False)
        else:
            estimator = MSM.MaximumLikelihoodHMSM(nstates=nstates, lag=lag)
        return estimator.estimate(traj)

    def HMSM_test_all(self, traj, nstates, lag):
        MLH_estimator = self.create_HMSM_model_estimator(traj, nstates, lag, False)
        print("Maximum Likelihood Estimator:\n", MLH_estimator)
        bayesian_estimator = self.create_HMSM_model_estimator(traj, nstates, lag, True)
        print("Bayesian Estimator:\n", bayesian_estimator)

########################################
# Functionality tests
########################################
transition_matrix = np.array([[0.25, 0.25, 0.5],
                                  [0.4, 0.2, 0.4],
                                  [0.15, 0.55, 0.3]])
pobs = transition_matrix
initial_states = [0]

test = MSM_Test()

test.MSM_simulate_test_all(transition_matrix, 3, 3, initial_states)
print("\n")
test.HMSM_simulate_test_all(transition_matrix, pobs, 3, 3, initial_states)

print("\n\n")
traj, obs = test.test_simulate_HMSM(transition_matrix, pobs, 100, 1, None)
print(traj)
test.HMSM_test_all(traj[0], 3, 1)
