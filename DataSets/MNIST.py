import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())   #Flatten the images! Could be done with numpy reshape
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))   #10 because dataset is numbers from 0 - 9

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)  # train the model


print(x_train[0])

plt.imshow(x_test[0],cmap=plt.cm.binary) #cmap changes the colour it displays in
plt.show()
model.save('test')
new_model = tf.keras.models.load_model('test')
predictions = new_model.predict(x_test)
import numpy as np
print(np.argmax(predictions[0]))
#print(y_train[0])
