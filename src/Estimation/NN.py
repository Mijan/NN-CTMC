import numpy as np
import tensorflow as tf
from typing import Tuple, List

class CTMCModel(tf.keras.Model):
    def __init__(self, base_model: tf.keras.Model, **kwargs):
        super(CTMCModel, self).__init__(**kwargs)
        self.base_model = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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





# code for debugging purposes
if __name__ == "__main__":
    from src.Models.models import BirthDeath
    from src.Models.models import ThreeSpeciesModel
    from src.Simulator.SSA import SSASimulator
    from src.Estimation.NN import CTMCModel
    from src.Models.utils import getReactionsForObservations

    dynamic_model = BirthDeath()
    # dynamic_model = ThreeSpeciesModel()
    simulator = SSASimulator(dynamic_model)

    parameters = dynamic_model.getDefaultParameter()
    y, t = simulator.run_ssa(np.array([5]), 100, parameters)
    # y, t = simulator.run_ssa(np.array([80000, 10, 10]), 100, parameters)
    num_states = y.shape[1]

    reaction_indices, unique_reaction_mapping = getReactionsForObservations(y, dynamic_model.getStoichiometry())
    num_unique_stoch = len(np.unique(unique_reaction_mapping))

    # test computeLossFunction
    true_alpha = np.array([dynamic_model.getPropensities(obs, time) for obs, time in zip(y, t)])
    true_alpha_unique = np.zeros(shape=(true_alpha.shape[0], num_unique_stoch))
    for rct, rct_map in enumerate(unique_reaction_mapping):
        true_alpha_unique[:, rct_map] += true_alpha[:, rct]

    Model = tf.keras.models.Sequential([
        tf.keras.layers.Input((num_states,)),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.selu),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.selu),
        tf.keras.layers.Dense(num_unique_stoch, activation=tf.keras.activations.softplus)
    ])

    custom_model = CTMCModel(Model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    custom_model.compile(optimizer)

    tf.config.experimental_run_functions_eagerly(True)
    train_dataset = tf.data.Dataset.from_tensor_slices((y[:-1], t[:-1], reaction_indices)).batch(
        256)  # Batch size of 32 as an example
    print(np.max(reaction_indices))
    custom_model.fit(train_dataset, epochs=10)

    # Step 1: Use the trained custom_model to predict propensities for the states saved in y
    nn_predictions = custom_model.predict(y)
    expected_propensities = true_alpha_unique

    comparison = nn_predictions - expected_propensities

    mse = np.mean(np.square(comparison))
    print(f"Mean Squared Error between NN predictions and expected propensities: {mse}")

    # Or simply compare them directly for a few instances:
    for i in range(10):  # just printing the first 10 instances for brevity
        print(f"NN Prediction: {nn_predictions[i]}, Expected: {expected_propensities[i]}, Difference: {comparison[i]}")


