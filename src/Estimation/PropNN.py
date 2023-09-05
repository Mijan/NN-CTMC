import numpy as np
import tensorflow as tf
from typing import Tuple, List
from tensorflow.keras.initializers import Constant

class FeedForwardPropensityModel(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs, num_layers = 4, num_neurons = 64, learning_rate=0.001, **kwargs):
        super(FeedForwardPropensityModel, self).__init__(**kwargs)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input((num_inputs,)))
        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(num_neurons, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.Dense(num_outputs, activation=tf.keras.activations.softplus,
                                        kernel_initializer=Constant(0.1)))

        self.base_model = model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=self.optimizer, loss=self.computeLossTF)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.base_model(inputs)

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> dict:
        y, t, reaction_indices = data

        with tf.GradientTape() as tape:
            alpha = self.base_model(y, training=True)
            loss = self.computeLossTF(alpha, t, reaction_indices)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the loss tracker
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [self.loss_tracker]

    def computeLossTF(self, alpha: tf.Tensor, times: tf.Tensor, reaction_indices: tf.Tensor) -> tf.Tensor:
        alpha = tf.cast(alpha, dtype=tf.float32)
        times = tf.cast(times, dtype=tf.float32)
        alpha_sum = tf.reduce_sum(alpha, axis=1)

        def mapped_fn(obs_nbr):
            reaction_index = reaction_indices[obs_nbr]
            times_rct = tf.gather(times, obs_nbr)
            alpha_sum_rct = tf.gather(alpha_sum, obs_nbr)
            temp_loss = alpha_sum_rct * times_rct - tf.math.log(tf.gather(alpha[:, reaction_index], obs_nbr))
            return tf.reduce_sum(temp_loss)

        loss_values = tf.map_fn(mapped_fn, tf.range(tf.shape(reaction_indices)[0]), dtype=tf.float32)
        return tf.reduce_sum(loss_values)



def getTrainDatasetFromSimulations(y: np.array, t: np.array, reaction_indices: np.array,  batch_size = 256):
    times_spent = t[1:] - t[:-1]
    train_dataset = tf.data.Dataset.from_tensor_slices((y[:-1], times_spent, reaction_indices)).batch(
        batch_size)
    return train_dataset


# code for debugging purposes
if __name__ == "__main__":
    from src.Models.models import BirthDeath
    from src.Models.models import ThreeSpeciesModel
    from src.Simulator.SSA import SSASimulator
    from src.Estimation.PropNN import FeedForwardPropensityModel
    from src.Estimation.PropNN import getTrainDatasetFromSimulations
    from src.Models.utils import getReactionsForObservations

    # dynamic_model = BirthDeath()
    dynamic_model = ThreeSpeciesModel()
    simulator = SSASimulator(dynamic_model)

    parameters = dynamic_model.getDefaultParameter()
    # y, t = simulator.run_ssa(np.array([5]), 100, parameters)
    y, t = simulator.run_ssa(np.array([80000, 10, 10]), 1000, parameters)
    num_states = y.shape[1]

    reaction_indices, unique_reaction_mapping = getReactionsForObservations(y, dynamic_model.getStoichiometry())
    num_unique_stoch = len(np.unique(unique_reaction_mapping))


    custom_model = FeedForwardPropensityModel(num_inputs=num_states, num_outputs=num_unique_stoch, num_layers=4)

    train_dataset = getTrainDatasetFromSimulations(y, t,reaction_indices)
    custom_model.fit(train_dataset, epochs=10)


