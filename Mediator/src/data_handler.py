import tensorflow as tf

class DataHandler:

    def load(self):

        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        return test_images, test_labels

if __name__ == "__main__":
    handler = DataHandler()
    handler.load()