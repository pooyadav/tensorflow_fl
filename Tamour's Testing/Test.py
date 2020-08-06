import collections
import keras

import numpy as np
np.random.seed(0)

import tensorflow_federated as tff


print(tff.federated_computation(lambda: 'Hello World')())