
import numpy as np
from typing import List


def computeMLETransitionMatrix(state_vector: List[np.array], time_vector: np.array) -> np.array:
    # Convert state sequences to unique integers for efficient indexing
    states, unique_indices = np.unique(np.vstack(state_vector), axis=0, return_inverse=True)

    n_states = len(states)

    # Initialize arrays for transition counts and state times
    transition_counts = np.zeros((n_states, n_states))
    state_times = np.zeros(n_states)

    # Count transitions and compute state times
    for idx in range(1, len(unique_indices)):
        transition_counts[unique_indices[idx - 1], unique_indices[idx]] += 1
        state_times[unique_indices[idx - 1]] += time_vector[idx] - time_vector[idx - 1]

    # Calculate MLE transition rates
    with np.errstate(divide='ignore', invalid='ignore'):
        mle_transition_matrix = transition_counts / state_times[:, np.newaxis]
    mle_transition_matrix[np.isnan(mle_transition_matrix)] = 0

    return mle_transition_matrix

if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    from src.Models.models import ThreeSpeciesModel
    from src.Estimation.MLE import computeMLETransitionMatrix
    from src.Models.models import ThreeSpeciesModel
    from src.Simulator.SSA import SSASimulator
    from src.Plotter.SystemPlotter import SysPlotter
    import os

    # data_folder = Path("../../data")
    # data_name = "ThreeState.npz"
    # print(os.getcwd())
    # model = ThreeSpeciesModel()
    # #
    # data = np.load(str(data_folder / data_name))
    # y = data['y']
    # t = data['t']

    model = ThreeSpeciesModel()
    simulator = SSASimulator(model)

    parameters = model.getDefaultParameter()

    y, t = simulator.run_ssa(np.array([80000, 10, 10]), 1000, parameters)
    plotter = SysPlotter()
    fig = plotter.plotSystem(y, t, model)

    model = ThreeSpeciesModel()
    simulator = SSASimulator(model)
    computeMLETransitionMatrix(y, t)


# def computeMLETransitionMatrix(state_vector: List[np.array], time_vector: np.array):
#     assert len(state_vector) == len(time_vector)
#     assert time_vector.ndim == 1
#
#     state_vector =  state_vector - np.min(state_vector)
#
#     n = max(state_vector) + 1
#     mat_ = np.zeros([n, n])  # matrix if nxn dim
#     time_ = np.zeros([n, n])
#
#     for (i, j, t0, t1) in zip(state_vector, state_vector[1:], time_vector, time_vector[1:]):
#         mat_[i,j] +=1
#         time_[i,j] += t1 - t0
#
#     total_time_spent_in_state = np.sum(time_, 1)
#
#     for i in range(n):
#         for j in range(n):
#             mat_[i, j] /= total_time_spent_in_state[i]
#
#     # populate the diagonal matrix
#     sum_rows = np.sum(mat_, 1)
#     for i in range(n):
#         mat_[i, i] = sum_rows[i]
#
#     return mat_