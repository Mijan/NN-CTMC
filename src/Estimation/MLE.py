
import numpy as np
from src.Models.paper_examples import Model
import matplotlib.pyplot as plt
from src.Utils.GeneralUtils import getRowsCols
from src.Estimation.utils import createPropensityPlot


class MLEstimator:
    def __init__(self, model: Model, observation_vector: np.array, time_vector: np.array):
        self.__observation_vector = observation_vector
        self.__num_observations = observation_vector.shape[0]

        assert self.__num_observations == len(time_vector)
        self.__time_vector = time_vector

        self.__model = model
        self.__stoich = model.getStoichiometry()
        self.__num_reactions = model.getNumReactions()

        self.__unique_states, self.__unique_indices = np.unique(observation_vector, axis=0, return_inverse=True)
        self.__num_unique_states = self.__unique_states.shape[0]

        self.__state_to_idx = {tuple(state): idx for idx, state in enumerate(self.__unique_states)}


    def computeMLETransitionMatrix(self) -> np.array:
        # Convert state sequences to unique integers for efficient indexing

        n_states = len(self.__unique_states)

        # Initialize arrays for transition counts and state times
        transition_counts = np.zeros((n_states, n_states))
        state_times = np.zeros(n_states)

        # Count transitions and compute state times
        for idx in range(1, len(self.__unique_indices)):
            transition_counts[self.__unique_indices[idx - 1], self.__unique_indices[idx]] += 1
            state_times[self.__unique_indices[idx - 1]] += self.__time_vector[idx] - self.__time_vector[idx - 1]

        # Calculate MLE transition rates
        with np.errstate(divide='ignore', invalid='ignore'):
            mle_transition_matrix = transition_counts / state_times[:, np.newaxis]
        mle_transition_matrix[np.isnan(mle_transition_matrix)] = 0

        return mle_transition_matrix

    def plotMLEstimates(self, mle_transition_matrix: np.array):
        estimated_propensities = self.__extractPropensitiesFromMLEMatrix(mle_transition_matrix)

        true_propensities = np.zeros((self.__num_observations, self.__num_reactions))
        for i, state in enumerate(self.__observation_vector):
            true_propensities[i] = self.__model.getPropensities(state, self.__time_vector[i])

        fig = createPropensityPlot(estimated_propensities, true_propensities)
        return fig

    def __extractPropensitiesFromMLEMatrix(self, mle_transition_matrix):
        # Initialize the estimated propensities matrix
        estimated_propensities = np.zeros((self.__num_observations, self.__num_reactions))

        for obs_nbr, obs_state in enumerate(self.__observation_vector):
            obs_state_idx = self.__state_to_idx[tuple(obs_state)]

            # Iterate through the stoichiometry to determine possible target states
            for reaction_idx, reaction_stoich in enumerate(self.__stoich):
                target_state = obs_state + reaction_stoich
                target_state_idx = self.__state_to_idx.get(tuple(target_state))

                # If target_state_idx is not None, it means the target state exists in the MLE matrix
                if target_state_idx is not None:
                    estimated_propensities[obs_nbr, reaction_idx] = mle_transition_matrix[
                        obs_state_idx, target_state_idx]

        return estimated_propensities

    def __createReactionMapping(self):

        # Convert stoichiometry matrix rows to strings to make them hashable.
        stoich_dict = {}
        for rct_nbr, stoich_vect in enumerate(self.__stoich):
            key = tuple(stoich_vect)
            stoich_dict.setdefault(key, []).append(rct_nbr)

        # Construct the reactions_mapping using only successive states
        reactions_mapping = {}
        for i in range(self.__num_observations - 1):  # -1 to avoid indexing out of range
            source_state = self.__observation_vector[i]
            target_state = self.__observation_vector[i + 1]

            source_idx = self.__unique_indices[i]
            target_idx = self.__unique_indices[i + 1]

            difference = tuple(target_state - source_state)
            matching_reactions = stoich_dict.get(difference, [])

            if matching_reactions:
                key = (source_idx, target_idx)
                reactions_mapping[key] = matching_reactions[0] if len(matching_reactions) == 1 else tuple(
                    matching_reactions)

        return reactions_mapping


# Debugging run
if __name__ == '__main__':
    from src.Models.paper_examples import BirthDeath
    from src.Models.paper_examples import ThreeSpeciesModel
    from src.Models.paper_examples import BirthDeathPaper
    from src.Simulator.SSA import SSASimulator
    from src.Estimation.MLE import MLEstimator
    # model = BirthDeath()
    model = BirthDeathPaper()
    # model = ThreeSpeciesModel()
    simulator = SSASimulator(model)

    parameters = model.getDefaultParameter()
    # y, t = simulator.run_ssa(np.array([5]), 10000, parameters)
    y, t = simulator.run_ssa(np.array([500]), 10000, parameters)
    # y, t = simulator.run_ssa(np.array([80000, 10, 10]), 500, parameters)
    estimator = MLEstimator(model, y, t)
    mle_matrix = estimator.computeMLETransitionMatrix()
    fig = estimator.plotMLEstimates(mle_matrix)
    fig.show()
