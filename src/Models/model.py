from abc import ABC, abstractmethod
import numpy as np
from typing import List


class Model(ABC):

    def __init__(self, species_names, parameter_names, model_name):
        super().__init__()
        self._species_names = species_names
        self._parameter_names = parameter_names
        self._parameters = np.array(len(parameter_names))
        self.model_name = model_name

    @abstractmethod
    def getPropensities(self, state: np.array, t: float) -> np.array:
        pass

    def getSpeciecByReaction(self) -> List[list]:
        species_by_reaction = []
        for rct_nbr in self.getNumReactions():
            species = [i for i in self.getNumSpecies()]
            species_by_reaction.append(species)
        return species_by_reaction

    @abstractmethod
    def getStoichiometry(self) -> np.array:
        pass

    @abstractmethod
    def getDefaultParameter(self) -> np.array:
        pass

    @abstractmethod
    def getDefaultInitialState(self) -> np.array:
        pass

    def setParameter(self, params: np.array) -> None:
        if (len(params) != self.getNumParameter()):
            raise Exception(
                f"Provided number of parameters: {len(params)}, but expected number of parameters {self.getNumParameter()}")
        self._parameters = params.copy()

    def getNumSpecies(self) -> int:
        return len(self._species_names)

    def getNumParameter(self) -> int:
        return len(self._parameter_names)

    def getNumReactions(self) -> int:
        stoich = self.getStoichiometry()
        return len(stoich)

    def getSpeciesNames(self):
        return self._species_names
