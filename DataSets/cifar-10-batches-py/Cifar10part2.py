import tensorflow as tf
import tensorflow_federated as tff
(x_train, y_train), (x_test, y_test) = tff.simulation.datasets.cifar100.load_data()

print('x_train shape:', x_train.shape)
