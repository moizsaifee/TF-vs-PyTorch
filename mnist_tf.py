import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import GradientTape
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.datasets import mnist

# Download the MNIST Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# Model Definition
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        output = self.d2(x)
        return output


# Instantiate Model, Optimizer, Loss
model = MyModel()
optimizer = Adam()
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='sum')

accuracy_object = SparseCategoricalAccuracy()

for epoch in range(2):
    # Reset the metrics at the start of the next epoch
    train_loss = 0
    train_n = 0
    for images, labels in train_ds:
        with GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            train_loss += loss.numpy()
            train_n += labels.shape[0]
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss /= train_n

    test_loss = 0
    test_accuracy = 0
    test_n = 0
    print(f'train loss: {train_loss}')
    for test_images, test_labels in test_ds:
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(test_images, training=False)
        t_loss = loss_object(test_labels, predictions)

        test_loss += t_loss.numpy()
        test_accuracy += accuracy_object(test_labels, predictions)*test_labels.shape[0]
        test_n += test_labels.shape[0]

    test_loss /= test_n
    test_accuracy /= test_n

    template = 'Epoch {}, Loss: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss,
                          test_loss,
                          test_accuracy * 100))