from src.utils.generalutils import getRowsCols
import matplotlib.pyplot as plt
import numpy as np

def createPropensityPlot( estimated_propensities, true_propensities):
    num_reactions = true_propensities.shape[1]
    nrows, ncols = getRowsCols(num_reactions)

    fig, axs = plt.subplots(nrows, ncols, figsize=( 5 *ncols, 5* nrows))
    for rct_nbr in range(num_reactions):
        if nrows > 1:
            ax = axs[rct_nbr // ncols, rct_nbr % ncols]  # Select the right subplot
        else:
            ax = axs[rct_nbr]

        # Sum true propensities for reactions with the same stoichiometry
        sum_true_propensities = true_propensities[:, rct_nbr]

        ax.scatter(sum_true_propensities, estimated_propensities[:, rct_nbr], alpha=0.5)
        ax.set_xlabel(f'True Propensities for Reaction {rct_nbr + 1}')
        ax.set_ylabel(f'Estimated Propensities for Reaction {rct_nbr + 1}')

        min_val = min(sum_true_propensities.min(), estimated_propensities[:, rct_nbr].min())
        max_val = max(sum_true_propensities.max(), estimated_propensities[:, rct_nbr].max())

        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        ax.grid(True)
        ax.set_title(f"Comparison for Reaction {rct_nbr + 1}")
    plt.tight_layout()
    return fig