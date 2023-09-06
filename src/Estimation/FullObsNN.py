import numpy as np
import tensorflow as tf
from typing import Tuple, List
from abc import ABC, abstractmethod
from tensorflow.keras.initializers import Constant
import time

class FullObsNN(tf.keras.Model, ABC):
    def __init__(self, learning_rate=1e-3, **kwargs):
        super(FullObsNN, self).__init__(**kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @abstractmethod
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        pass


    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> dict:
        y, t, reaction_indices = data

        with tf.GradientTape() as tape:
            alpha = self.call(y, training=True)
            loss = self.computeLossTF(alpha, t, reaction_indices)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [self.loss_tracker]

    def computeLossTF(self, alpha: tf.Tensor, times: tf.Tensor, reaction_indices: tf.Tensor) -> tf.Tensor:
        alpha = tf.cast(alpha, dtype=tf.float32)
        alpha_clipped = tf.clip_by_value(alpha, 1e-32, tf.float32.max)
        times = tf.cast(times, dtype=tf.float32)
        alpha_sum = tf.reduce_sum(alpha_clipped, axis=1, keepdims=True)

        alpha_gathered = tf.gather(alpha_clipped, reaction_indices, axis=1, batch_dims=1)

        times = tf.reshape(times, [-1, 1])
        alpha_sum = tf.reshape(alpha_sum, [-1, 1])
        temp_loss = alpha_sum * times - tf.math.log(alpha_gathered)
        loss_values = tf.reduce_sum(temp_loss, axis=0)
        loss = tf.reduce_sum(loss_values)
        return loss


class CombinedReactionsNN(FullObsNN):
    def __init__(self, num_inputs, num_outputs, num_layers=4, num_neurons=64, **kwargs):
        super(CombinedReactionsNN, self).__init__(**kwargs)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input((num_inputs,)))
        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(num_neurons))
            model.add(tf.keras.layers.Activation(tf.keras.activations.selu))
        model.add(tf.keras.layers.Dense(num_outputs, kernel_initializer=Constant(0.1)))
        model.add(tf.keras.layers.Activation(tf.keras.activations.softplus))

        self.base_model = model
        self.compile(optimizer=self.optimizer, loss=self.computeLossTF)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.base_model(inputs, training = training)


class IndividualReactionsNN(FullObsNN):
    def __init__(self, num_outputs, inputs_by_output, num_layers=4, num_neurons=64, **kwargs):
        assert len(inputs_by_output) == num_outputs

        super(IndividualReactionsNN, self).__init__(**kwargs)
        self.inputs_by_outputs = inputs_by_output

        models = []
        for inputs_for_output in inputs_by_output:
            num_inputs_for_output = len(inputs_for_output) if len(inputs_for_output) > 0 else 1

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input((num_inputs_for_output,)))
            for _ in range(num_layers):
                model.add(tf.keras.layers.Dense(num_neurons, activation=tf.keras.activations.selu))
            model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.softplus,
                                            kernel_initializer=Constant(0.1)))
            models.append(model)

        self.models = models
        self.compile(optimizer=self.optimizer, loss=self.computeLossTF)

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        alpha = []
        for output_nbr, inputs_for_output in enumerate(self.inputs_by_output):
            if len(inputs_for_output) >= 1:
                alpha[output_nbr] = self.models[output_nbr](inputs[inputs_for_output, :], training=True)
            else:
                alpha[output_nbr] = self.models[output_nbr](1, training=training)

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        alpha = []
        for output_nbr, inputs_for_output in enumerate(self.inputs_by_outputs):
            if len(inputs_for_output) >= 1:
                sliced_inputs = tf.gather(inputs, inputs_for_output, axis=1)
                alpha.append(self.models[output_nbr](sliced_inputs, training=training))
            else:
                dummy_input = tf.ones((tf.shape(inputs)[0], 1))
                alpha.append(self.models[output_nbr](dummy_input, training=training))

        return tf.concat(alpha, axis=1)


class DataPreparatorFullObs():
    def __init__(self, batch_size=256):
        self.__trajs = []
        self.__times_spent = []
        self.__reaction_indices = []
        self.__batch_size = batch_size

    def addTrajectory(self, y: np.array, t: np.array, reaction_indices: np.array):
        assert len(y) == len(t)
        assert len(reaction_indices) == len(y) - 1

        times_spent = t[1:] - t[:-1]
        self.__trajs.extend(y[:-1])
        self.__times_spent.extend(times_spent)
        self.__reaction_indices.extend(reaction_indices)

    def getTraindDataset(self):
        trajs_np = np.array(self.__trajs)
        times_spent_np = np.array(self.__times_spent)
        reaction_indices_np = np.array(self.__reaction_indices)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (trajs_np, times_spent_np, reaction_indices_np)
        ).batch(self.__batch_size)
        return train_dataset


# code for debugging purposes
if __name__ == "__main__":
    from src.Simulator.SSA import SSASimulator
    from src.Models.example_networks import LacGfp
    import matplotlib.pyplot as plt
    from src.Simulator.SSA import SSASimulator
    import numpy as np
    from src.Estimation.FullObsNN import CombinedReactionsNN
    from src.Estimation.FullObsNN import DataPreparatorFullObs
    from src.Models.utils import getReactionsForObservations
    from src.Estimation.utils import createPropensityPlot
    from src.Estimation.MLE import MLEstimator

    model_lac = LacGfp()
    simulator_lac = SSASimulator(model_lac)

    parameters = model_lac.getDefaultParameter()
    x0 = model_lac.getDefaultInitialState()
    y, t = simulator_lac.run_ssa(x0, 10, parameters)
    num_states = y.shape[1]

    reaction_indices, unique_reaction_mapping = getReactionsForObservations(y, model_lac.getStoichiometry())
    num_unique_stoch = len(np.unique(unique_reaction_mapping))

    custom_model = CombinedReactionsNN(num_inputs=num_states, num_outputs=num_unique_stoch, num_layers=4)
    data_preparator = DataPreparatorFullObs()
    data_preparator.addTrajectory(y, t, reaction_indices)

    num_trajs = 50
    for num_traj in range(num_trajs - 1):
        y, t = simulator_lac.run_ssa(x0, 10, parameters)
        reaction_indices, unique_reaction_mapping = getReactionsForObservations(y, model_lac.getStoichiometry())
        data_preparator.addTrajectory(y, t, reaction_indices)

    train_dataset = data_preparator.getTraindDataset()

    custom_model.fit(train_dataset, epochs=200)

