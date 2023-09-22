import tensorflow as tf
from typing import List
import numpy as np


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

    # def train(self, epochs: int):
    #
    #     for model, dataset, optimizer, loss_tracker, loss in zip(self.models,
    #                                                              self.datasets,
    #                                                              self.optimizers,
    #                                                              self.loss_trackers,
    #                                                              self.losses):
    #
    #         for epoch in range(epochs):
    #             for step, (x_batch, y_batch) in enumerate(dataset):
    #
    #                 with tf.GradientTape() as tape:
    #                     preds = model(x_batch, training=True)
    #                     loss_value = loss(y_batch, preds)
    #
    #                 grads = tape.gradient(loss_value, model.trainable_variables)
    #                 optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #
    #                 loss_tracker.update_state(loss_value)
    #
    #             print(
    #                 "Epoch{}---Loss--------{}".format(epoch, float(loss_tracker.result()))
    #             )
    #             loss_tracker.reset_states()

    def compile_and_fit(self, epochs):

        for model, dataset, optimizer in zip(self.models,
                                             self.datasets,
                                             self.optimizers):

            model.compile(loss='mae',
                          optimizer=optimizer)

            model.fit(dataset, epochs=epochs)


class Generator:
    def __init__(self,
                 input_width,
                 label_width,
                 train_df,
                 test_df):
        self.train_df = train_df
        self.test_df = test_df

        self.input_width = input_width
        self.label_width = label_width

        self.total_widow_size = input_width + label_width

    def split_window(self, features):
        inputs = features[:, :self.input_width, :]
        labels = features[:, self.total_widow_size-1:, -1:]

        return inputs, labels

    def make_generator(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_widow_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    def train_generator(self):
        return self.make_generator(self.train_df)

    def test_generator(self):
        return self.make_generator(self.test_df)





