import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28

#The tf.data.experimental.CsvDataset class can be used to read csv records directly
# from a gzip file with no intermediate decompression step.
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

#That csv reader class returns a list of scalars for each record.
# The following function repacks that list of scalars into a (feature_vector, label) pair.
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label

#TensorFlow is most efficient when operating on large batches of data.

#So instead of repacking each row individually make a new Dataset that takes batches of 10000-examples,
# applies the pack_row function to each batch, and then splits the batches back up into individual records:

packed_ds = ds.batch(10000).map(pack_row).unbatch()

#Takes the first feature labels
for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)

#To keep this tutorial relatively short use just the first 1000 samples for validation,
# and the next 10 000 for training:
N_VALIDATION = int(1000)
N_TRAIN = int(10000)
BUFFER_SIZE = int(10000)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

#The Dataset.skip and Dataset.take methods make this easy.

#At the same time, use the Dataset.cache method to ensure that the loader
# doesn't need to re-read the data from the file on each epoch:
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

#These datasets return individual examples.
# Use the .batch method to create batches of an appropriate size for training.
# Before batching also remember to .shuffle and .repeat the training set.
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
#Start with a simple model using only layers.Dense as a baseline, then create larger versions, and compare them.
#Training procedure
#Many models train better if you gradually reduce the learning rate during training.
# Use optimizers.schedules to reduce the learning rate over time:
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

#The code above sets a schedules.InverseTimeDecay to hyperbolically decrease the learning rate
# to 1/2 of the base rate at 1000 epochs, 1/3 at 2000 epochs and so on.

step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

#Each model in this tutorial will use the same training configuration.
# So set these up in a reusable way, starting with the list of callbacks.

#The training for this tutorial runs for many short epochs.
# To reduce the logging noise use the tfdocs.
# EpochDots which simply prints a . for each epoch, and a full set of metrics every 100 epochs.

#Next include callbacks.EarlyStopping to avoid long and unnecessary training times.
# Note that this callback is set to monitor the val_binary_crossentropy, not the val_loss.
# This difference will be important later.

#Use callbacks.TensorBoard to generate TensorBoard logs for the training.

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

#Similarly each model will use the same Model.compile and Model.fit settings:

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

size_histories = {}

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

"""
small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')



medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])

size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")
#medium is defo overtrained


large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
size_histories['large'] = compile_and_fit(large_model, "sizes/large")
"""

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()

#You need less epochs the more bigger the model is

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()


regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']
#Occam's Razor principle: given two explanations for something,
# the explanation most likely to be correct is the "simplest" one,
# the one that makes the least amount of assumptions.
#Thus a common way to mitigate overfitting is to put constraints on the complexity
# of a network by forcing its weights only to take small values

"""

    L1 regularization, where the cost added is proportional to the absolute value of the weights coefficients (i.e. to what is called the "L1 norm" of the weights).

    L2 regularization, where the cost added is proportional to the square of the value of the weights coefficients (i.e. to what is called the squared "L2 norm" of the weights). L2 regularization is also called weight decay in the context of neural networks. Don't let the different name confuse you: weight decay is mathematically the exact same as L2 regularization.
"""

l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()

#As you can see, the "L2" regularized model is now much more competitive with the
# the "Tiny" model. This "L2" model is also much more resistant to overfitting
# than the "Large" model it was based on despite having the same number of parameters.
"""The "dropout rate" is the fraction of the features that are being zeroed-out; 
it is usually set between 0.2 and 0.5. 
At test time, no units are dropped out, and instead the layer's output values 
are scaled down by a factor equal to the dropout rate, 
so as to balance for the fact that more units are active than at training time."""

dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()