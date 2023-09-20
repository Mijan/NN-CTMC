from .model import Model
import numpy as np
from typing import List


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

    def getDefaultInitialState(self) -> np.array:
        return np.array([0])

    def getSpeciecByReaction(self) -> List[list]:
        return [[], [0]]


class LacGfp(Model):

    def __init__(self):
        super().__init__(["lacI", "LACI", "LACI2", "PLac", "O2Lac", "O4Lac", "gfp", "GFP", "mGFP"],
                         ["theta_1", "theta_2", "theta_3", "theta_4", "theta_5", "theta_6", "theta_7", "theta_8",
                          "theta_9", "theta_10", "theta_11", "theta_12", "theta_13", "theta_14", "theta_15", "theta_16",
                          "theta_17", "theta_18", "IPTG"], "Lac-Gfp Model")

    def getPropensities(self, state: np.array, t: float) -> np.array:
        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10, theta_11, theta_12, theta_13, theta_14, theta_15, theta_16, theta_17, theta_18, IPTG = self._parameters[
                                                                                                                                                                                          :19]
        lacI, LACI, LACI2, PLac, O2Lac, O4Lac, gfp, GFP, mGFP = state[:9]

        prop1 = theta_1
        prop2 = theta_2 * lacI
        prop3 = theta_3 * lacI
        prop4 = LACI * (theta_4 + theta_5 * IPTG)
        prop5 = theta_6 * LACI * (LACI - 1)
        prop6 = theta_7 * LACI2
        prop7 = theta_8 * LACI2 * PLac
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

        return np.array(
            [prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8, prop9, prop10, prop11, prop12, prop13, prop14,
             prop15, prop16, prop17, prop18])

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

    def getDefaultInitialState(self) -> np.array:
        return np.array([3, 7, 0, 0, 0, 62, 0, 0, 0])

    def getSpeciecByReaction(self) -> List[list]:
        return [
            [], [0], [0], [1], [1], [2], [2, 3], [4], [4], [5], [3, 4, 5],[6], [6], [7], [7], [8]]


class ERK(Model):

    def __init__(self):
        super().__init__(
            ["Ras", "Ras_star", "Raf", "Raf_star", "Mek", "Mek_star", "Erk", "Erk_star", "Nfb", "Nfb_star", "FgfR",
             "FgfR_star", "H", "H_F", "H_F_R"],
            ["k_1_2", "K_1_2", "k_2_1", "K_2_1", "k_3_4", "K_3_4", "K_NFB", "k_4_3", "K_4_3", "k_5_6", "K_5_6", "k_6_5",
             "K_6_5", "k_7_8", "K_7_8", "k_8_7", "K_8_7", "f_1_2", "F_1_2", "f_2_1", "F_2_1", "h_nfb", "r_h", "r_3_4",
             "r_4_3", "r_5_6", "r_6_5", "r_5_7", "r_7_5"], "Lac-Gfp Model")

    def getPropensities(self, state: np.array, t: float) -> np.array:
        fgf_input = 5
        k_1_2, K_1_2, k_2_1, K_2_1, k_3_4, K_3_4, K_NFB, k_4_3, K_4_3, k_5_6, K_5_6, k_6_5, K_6_5, k_7_8, K_7_8, k_8_7, K_8_7, f_1_2, F_1_2, f_2_1, F_2_1, h_nfb, r_h, r_3_4, r_4_3, r_5_6, r_6_5, r_5_7, r_7_5 =        self._parameters[:]
        Ras, Ras_star, Raf, Raf_star, Mek, Mek_star, Erk, Erk_star, Nfb, Nfb_star, FgfR, FgfR_star, H, H_F, H_F_R = state[
                                                                                                                    :15]

        prop1 = k_1_2 * (r_h * H_F_R + (1 - r_h) * FgfR_star) * (Ras / (K_1_2 + Ras))
        prop2 = k_2_1 * (Ras_star / (K_2_1 + Ras_star))
        prop3 = k_3_4 * Ras_star * (Raf / (K_3_4 + Raf)) * (K_NFB ** h_nfb / (K_NFB ** h_nfb + Nfb_star ** h_nfb))
        prop4 = k_4_3 * (Raf_star / (K_4_3 + Raf_star))
        prop5 = k_5_6 * Raf_star * (Mek / (K_5_6 + Mek))
        prop6 = k_6_5 * (Mek_star / (K_6_5 + Mek_star))
        prop7 = k_7_8 * Mek_star * (Erk / (K_7_8 + Erk))
        prop8 = k_8_7 * (Erk_star / (K_8_7 + Erk_star))
        prop9 = f_1_2 * Erk_star * (Nfb / (F_1_2 + Nfb))
        prop10 = f_2_1 * (Nfb_star / (F_2_1 + Nfb_star))
        prop11 = r_3_4 * fgf_input * FgfR
        prop12 = r_4_3 * FgfR_star
        prop13 = r_5_6 * H * fgf_input
        prop14 = r_6_5 * H_F
        prop15 = r_5_7 * H_F * FgfR
        prop16 = r_7_5 * H_F_R

        return np.array(
            [prop1, prop2, prop3, prop4, prop5, prop6, prop7, prop8, prop9, prop10, prop11, prop12, prop13, prop14,
             prop15, prop16])

    def getStoichiometry(self) -> np.array:
        return np.array([
            [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -1]])

    def getDefaultParameter(self) -> np.array:
        return np.array(
            [0.71787774, 0.02358723, 0.80829515, 0.21497361, 4.35903194, 0.05998804, 0.03087217, 0.39595766, 0.45148145,
             2.46185222, 0.12292634, 0.68642178, 0.23168421, 1.86059625, 0.18694515, 2.69795479, 0.73323282, 0.04401609,
             0.06329493, 0.37816525, 0.84081636, 0.93105391, 0.95446949, 0.05624321, 0.35055068, 0.33902887, 0.1367214,
             3.29629793, 1.57101515])

    def getDefaultInitialState(self) -> np.array:
        return np.array([10, 0, 5, 0, 5, 0, 30, 0, 15, 0, 15, 0, 13, 0, 0])

    def getSpeciecByReaction(self) -> List[list]:
        return [[14, 11, 0], [1], [1, 2, 9], [3], [3, 4], [5], [5, 6], [7], [7, 8], [9], [10], [11], [12], [13],
                [13, 10], [14]]
