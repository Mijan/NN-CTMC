from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

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

  @abstractmethod
  def getStoichiometry(self) -> np.array:
      pass

  @abstractmethod
  def getDefaultParameter(self) -> np.array:
      pass

  def setParameter(self, params : np.array) -> None:
    if(len(params) != self.getNumParameter()):
      raise Exception(f"Provided number of parameters: {len(params)}, but expected number of parameters { self.getNumParameter()}")
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


class BirthDeathPaper(Model):

  def __init__(self):
      super().__init__(["X"], [], "Birth-Death Model (Paper)")

  def getPropensities(self, state: np.array, t: float) -> np.array:
      s = (t / 365.24) % 1
      s = np.round(s, decimals=1)

      birth_rate = np.abs(2.1 * np.cos(np.pi * s))
      death_rate = np.abs(2 * np.sin(np.pi * s))
      return np.array([birth_rate, death_rate])

  def getStoichiometry(self) -> np.array:
      return np.array([
          [1],
          [-1]])


  def getDefaultParameter(self) -> np.array:
      return np.array([])


class BirthDeath(Model):

  def __init__(self):
      super().__init__(["X"], ["k", "gamma"], "Birth-Death Model")

  def getPropensities(self, state: np.array, t: float) -> np.array:
      k, gamma = self._parameters[:2]
      X = state[0]
      return np.array([k, gamma * X])

  def getStoichiometry(self) -> np.array:
      return np.array([
          [1],
          [-1]])

  def getDefaultParameter(self) -> np.array:
      return np.array([1, 0.1])


class ThreeSpeciesModel(Model):

  def __init__(self):
    super().__init__(["Prey", "Preditor","SuperPreditor"],
                     ["alpha_1","alpha_2","alpha_3","alpha_4","alpha_5","alpha_6","alpha_7","alpha_8","alpha_9"],
                     "3 Species Model")

  def getPropensities(self, state: np.array, t: float) -> np.array:
    if len(state) != self.getNumSpecies():
      raise Exception(f"Provided number of states: {len(state)}, but expected number of states {len(self.getSpeciesNum())}")

    A, B, C = state[:3]
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = self._parameters[:9]

    N = 10**5
    alpha_1 = B*k1
    alpha_2 = k2
    alpha_3 = (A* k3) / N
    alpha_4 = k4
    alpha_5 = C * k5
    alpha_6 = k6
    alpha_7 = (A * k7) / N
    alpha_8 = (A * k8 * np.sqrt(B)) / N
    alpha_9 = k9 * np.log(B * C + 1)

    return np.array([alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7,alpha_8,alpha_9])

  def getStoichiometry(self) -> np.array:
      return np.array([
          [0, -1, 0],
          [0, 1, 0],
          [-1, 0, 0],
          [1, 0, 0],
          [0, 0, -1],
          [0, 0, 1],
          [1, 0, 0],
          [-1, 1, 0],
          [0, -1, 1]
      ])

  def getDefaultParameter(self) -> np.array:
      return np.array([0.5, 1.7, 3.9, 4.6, 2.7, 1.9, 6.1, 2.4, 1.5])