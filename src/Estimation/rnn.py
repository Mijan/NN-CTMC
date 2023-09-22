import tensorflow as tf
from typing import List


class MultiRNNTrainer:
    def __init__(self, datasets: List[tf.data.Dataset]):

        self.datasets = datasets
        self.models = []
        for _ in range(len(self.datasets)):
            self.models.append(tf.keras.models.Sequential([
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dense(units=1)
            ]))

        self.optimizers = [tf.keras.optimizers.Adam() for _ in range(len(self.models))]
        self.loss_trackers = [tf.keras.metrics.Mean(name="loss") for _ in range(len(self.models))]

        self.losses = [tf.losses.mean_squared_error for _ in range(len(self.models))]

        print(self.datasets)

    def train(self, epochs: int):

        for model, dataset, optimizer, loss_tracker, loss in zip(self.models,
                                                                 self.datasets,
                                                                 self.optimizers,
                                                                 self.loss_trackers,
                                                                 self.losses):

            for epoch in range(epochs):
                for step, (x_batch, y_batch) in enumerate(dataset):

                    with tf.GradientTape() as tape:
                        preds = model(x_batch, training=True)
                        loss_value = loss(y_batch, preds)

                    grads = tape.gradient(loss_value, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                    loss_tracker.update_state(loss_value)

                print(
                    "Epoch{}---Loss--------{}".format(epoch, float(loss_tracker.result()))
                )
                loss_tracker.reset_states()


# class CombinedRNNTrainer(tf.keras.Model):


