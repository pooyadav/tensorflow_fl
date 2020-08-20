import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds


#Download the imdb dataset
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

print(train_examples_batch)
print(train_labels_batch)

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#This embedding model can take a batch of sentences in a 1-D tensor of strings as input.
#The module preprocesses its input by splitting on spaces.
#Vocabulary
#Vocabulary contains 20,000 tokens and 1 out of vocabulary bucket for unknown tokens.

hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

print(train_examples_batch[0])

