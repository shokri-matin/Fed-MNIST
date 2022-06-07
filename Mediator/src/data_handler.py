import tensorflow as tf

class DataHandler:

    def load(self):

        mnist = tf.keras.datasets.mnist
        path = "E:\Fed-MNIST\mnist.npz"
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data(path)
        return test_images, test_labels

if __name__ == "__main__":
    handler = DataHandler()
    handler.load()