# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
#importing fashion dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""
#Above is: showing the images and the labels

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    #The first layer in this network, tf.keras.layers.Flatten,
    # transforms the format of the images from a two-dimensional array
    # (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
    # Think of this layer as unstacking rows of pixels in the image and lining them up.
    # This layer has no parameters to learn; it only reformats the data.
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    #The second (and last) layer returns a logits array with length of 10.
    # Each node contains a score that indicates the current image belongs to one of the 10 classes.
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""
Loss function — This measures how accurate the model is during training. 
You want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. 
The following example uses accuracy, the fraction of the images that are correctly classified.
"""
model.fit(train_images, train_labels, epochs=10)
#Fitting the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#Testing the accuracy of the model
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

for i in range(50):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
    #Setting up a way to see the image
