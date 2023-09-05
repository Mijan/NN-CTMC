
import numpy as np
from model import Model



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



class ThreeSpeciesModel(Model):

  def __init__(self):
    super().__init__(["Prey", "Preditor","SuperPreditor"],
                     ["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9"],
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

class ChemicalReactionNetwork(Model):

  def __init__(self):
      super().__init__(["A", "B"],
                       ["a1", "a2", "a3", "a4", "e1", "e2", "e3", "e4", "T"],
                       "Chemical Reaction Network")

  def getPropensities(self, state: np.array, t: float) -> np.array:
      if len(state) != self.getNumSpecies():
          raise Exception(
              f"Provided number of states: {len(state)}, but expected number of states {len(self.getSpeciesNum())}")

      A, B = state[:2]
      a1, a2, a3, a4, e1, e2, e3, e4, T = self._parameters[:9]
      R = 8.314
      k1 = a1 * np.exp(-e1/ (R*T))
      k2 = a2 * np.exp(-e2/ (R*T))
      k3 = a3 * np.exp(-e3/ (R*T))
      k4 = a4 * np.exp(-e4/ (R*T))

      alpha_1 = A * (A-1) * k1
      alpha_2 = A * B * k2
      alpha_3 = k3
      alpha_4 = k4

      return np.array([alpha_1, alpha_2, alpha_3, alpha_4])

  def getStoichiometry(self) -> np.array:
      return np.array([
          [-2, 0],
          [-1, -1],
          [1, 0],
          [0, 1]
      ])

  def getDefaultParameter(self) -> np.array:
      return np.array([630000, 770000, 5380000, 2240000, 39000, 36000, 40000, 40000, 273])