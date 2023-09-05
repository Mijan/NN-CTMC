from model import Model
import numpy as np

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


class LacGfp(Model):

  def __init__(self):
      super().__init__(["lacI", "LACI", "LACI2", "PLac", "O2Lac", "O4Lac", "gfp", "GFP", "mGFP"], ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6", "theta_7", "theta_8", "theta_9", "theta_10", "theta_11", "theta_12", "theta_13", "theta_14", "theta_15", "theta_16", "theta_17", "theta_18", "IPTG"], "Lac-Gfp Model")

  def getPropensities(self, state: np.array, t: float) -> np.array:
      theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10, theta_11, theta_12, theta_13, theta_14, theta_15, theta_16, theta_17, theta_18, IPTG = self._parameters[:19]
      lacI, LACI, LACI2, PLac, O2Lac, O4Lac, gfp, GFP, mGFP = state[:9]

      prop1 = theta_1
      prop2 = theta_2 * lacI
      prop3 = theta_3 * lacI
      prop4 = LACI * (theta_4 + theta_5 * IPTG)
      prop5 = theta_6 * LACI * (LACI-1)
      prop6 = theta_7 * LACI2
      prop7 =  theta_8 * LACI2 * PLac
      prop8 = theta_9 * O2Lac
      prop9 = theta_10 * O2Lac * (O2Lac - 1)
      prop10 = theta_11 * O4Lac
      prop11 = theta_12 * PLac
      prop12 = theta_13 * O2Lac
      prop13 = theta_14 * O4Lac
      prop14 = theta_15 * gfp
      prop15 = theta_16 * gfp
      prop16 = theta_17 * GFP
      prop17 = theta_18 * GFP
      prop18 = theta_17 * mGFP

      return np.array([prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8, prop9, prop10, prop11, prop12, prop13, prop14, prop15, prop16, prop17, prop18])

  def getStoichiometry(self) -> np.array:
      return np.array([
          [1, 0, 0, 0, 0, 0, 0, 0, 0],
          [-1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, -1, 0, 0, 0, 0, 0, 0, 0],
          [0, -2, 1, 0, 0, 0, 0, 0, 0],
          [0, 2, -1, 0, 0, 0, 0, 0, 0],
          [0, 0, -1, -1, 1, 0, 0, 0, 0],
          [0, 0, 1, 1, -1, 0, 0, 0, 0],
          [0, 0, 0, 0, -2, 1, 0, 0, 0],
          [0, 0, 0, 0, 2, -1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, -1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, -1, 0],
          [0, 0, 0, 0, 0, 0, 0, -1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, -1]])

  def getDefaultParameter(self) -> np.array:
      return np.array([1.5, 7.5, 1.5, 4.5, 5, 1650, 6, 0.48, 0.5, 230, 0.4, 125, 0.2, 0.01, 1.5, 32, 1, 2.2, 10])

