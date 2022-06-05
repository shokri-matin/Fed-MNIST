import tensorflow as tf
import numpy as np

class DataHandler:

    def __init__(self, username=2,  batch_size=32, number_of_parties=2) -> None:
        self.batch_size = batch_size
        self.number_of_parties = number_of_parties
        self.username = username
        self.mapper = {1:[0,29999], 2:[30000,60000]}
        self.tr_x, self.tr_y = self.load()

    def load(self):

        mnist = tf.keras.datasets.mnist
        (self.train_images, self.train_labels), (test_images, test_labels) = mnist.load_data()

        tr_x = self.train_images[self.mapper[self.username][0]: self.mapper[self.username][1]]
        tr_y = self.train_labels[self.mapper[self.username][0]: self.mapper[self.username][1]]

        return tr_x, tr_y

    def batch(self):
        index = np.random.choice(self.tr_x.shape[0], self.batch_size, replace=False)
        xtr = self.tr_x[index]
        ytr = self.tr_y[index]
        return xtr, ytr


if __name__ == "__main__":
    handler = DataHandler()
    handler.load()