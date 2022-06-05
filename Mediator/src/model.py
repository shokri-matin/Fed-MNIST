import tensorflow as tf

class Model:

    def __init__(self) -> None:

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model = self.create_model()

    def create_model(self, pretrained_weights = None, learning_rate = 0.01, shape = (28, 28)):

        model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=shape),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10)
                ])

        model.compile(optimizer=self.optimizer,
                        loss=self.loss_fn ,
                        metrics=['accuracy'])

        model.summary()

        if(pretrained_weights):
            model.load_weights(pretrained_weights)

        self.model = model
        return model

    def get_weights(self):
        weights = self.model.get_weights()
        return weights

    def backpropagate(self, grads):
        pass

    def feedforward(self, x):
        return self.model(x, training=True)

    def evaluate(self, x, y):
        test_loss, test_acc = self.model.evaluate(x,  y, verbose=2)
        return test_loss, test_acc