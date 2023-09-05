
from src.Models.paper_examples import Model
from typing import List
import numpy as np
import matplotlib.pylab as plt
from src.Utils.GeneralUtils import getRowsCols

class SysPlotter:
    def plotSystem(self, states: np.array, ts: list, model:Model):
        species_names = model.getSpeciesNames()
        num_species = len(species_names)

        nrows, ncols = getRowsCols(num_species)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

        if num_species >1:
            axs = axs.flatten()

        for state_nbr, state_name in enumerate(species_names):
            if num_species>1:
                ax = axs[state_nbr]
            else:
                ax = axs
            ax.set_xlabel('time')
            ax.set_ylabel(state_name)
            ax.plot(ts, states[:, state_nbr], label=state_name)

        fig.tight_layout()
        return fig