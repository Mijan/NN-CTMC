import tensorflow as tf
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


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

    def run_inference(self, test_datasets: List[tf.data.Dataset]):
        assert(len(test_datasets) == len(self.datasets), "check your test datasets")

        predictions = {}
        for test_dataset, model in zip(test_datasets, self.models):
            pred=[]
            targ=[]
            for feat, labels in test_dataset:
                pred.append(model(feat))
                targ.append(labels)
            pred = np.vstack(pred).reshape(-1)
            targ = np.vstack(targ).reshape(-1)
            # pred.reshape(targ.shape)
            predictions[model.name] = {"targ": targ, "pred" :pred}

        return predictions

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


def split_dataset(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()
    # assert("time" in dataframe.columns, "columns should have a time column")
    dataframe.set_index("time", inplace=True)

    columns = dataframe.columns
    num_columns = len(columns)
    num_splits = 1 / num_columns

    dataframe['t_sin'] = np.sin(2 * np.pi * dataframe.index)
    dataframe['t_cos'] = np.cos(2 * np.pi * dataframe.index)

    new_columns = ["t_sin", "t_cos"] + list(columns)
    dataframe = dataframe[new_columns]

    dataframes = []
    for i in range(0, num_columns):
        dataframes.append(
            dataframe.iloc[int(i * num_splits * len(dataframe)):int(len(dataframe) * (i + 1) * num_splits),
            [0, 1, i + 2]])

    i = 0
    dataframe_dict = {}
    for dataframe_ in dataframes:
        train_data, test_data = train_test_split(dataframe_, test_size=0.3, shuffle=False)
        dataframe_dict[f"dataframe_{i}"] = {"train_data": train_data, "test_data": test_data}
        i += 1

    return dataframe_dict


