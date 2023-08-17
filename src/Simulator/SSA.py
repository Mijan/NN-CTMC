
from src.Models.models import Model
import numpy as np

class SSASimulator():
    def __init__(self, model : Model):
        self._model = model

    def run_ssa(self, x0: np.array, T: float, params : np.array, save_props = False):
        t = 0
        x = x0

        states = []
        times = []

        self._model.setParameter(params)

        stoich = self._model.getStoichiometry()

        if(save_props):
            props_list = []

        while t < T:
            states.append(x.copy())
            times.append(t)

            props = self._model.getPropensities(x, t)

            tau = self.__getNextFiringTime(props)
            rct_nbr = self.__getFiringReaction(props)

            t += tau
            x = x + stoich[rct_nbr]
            if(save_props):
                props_list.append(props)


        return np.array(states), np.array(times)

    def __getFiringReaction(self, props):
        rct_nbr = np.random.choice(np.arange(len(props)), p=props/sum(props))
        return rct_nbr

    def __getNextFiringTime(self, props):

        total_props = sum(props)
        r_1 = np.random.uniform()
        tau = 1 / total_props * np.log(1 / r_1)
        return tau


if __name__ == '__main__':
    import numpy as np
    from src.Estimation.MLE import computeMLETransitionMatrix
    from src.Simulator.SSA import SSASimulator
    from src.Plotter.SystemPlotter import SysPlotter

    from src.Models.models import ThreeSpeciesModel

    model = ThreeSpeciesModel()
    simulator = SSASimulator(model)

    parameters = model.getDefaultParameter()
    y, t = simulator.run_ssa(np.array([10 ** 5, 10, 10]), 100, parameters)

    plotter = SysPlotter()
    fig = plotter.plotSystem(y, t, model)