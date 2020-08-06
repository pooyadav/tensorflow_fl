import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#Image classification in tensorflow
data = keras.datasets.fashion_mnist #Dummy dataset

(train_images, train_labels), (test_images, test_labels) = data.load_data()
#print(train_labels[1])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0
#Out of 1 makes it easier to work with machine learning

#print(train_images[1])
#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    #Flattened input layers for the first layer of neurons
    keras.layers.Dense(128, activation="relu"),
    #Fully connected layer, each neuron is connected to each other node,
    keras.layers.Dense(10, activation="softmax"),
    #Final layer, output layer, softmax pick values for each neuron so each value adds up to one

])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#Adam [1] is an adaptive learning rate optimization algorithm that’s been designed
# specifically for training deep neural networks.
#Loss: A scalar value that we attempt to minimize during our training of the model.
# The lower the loss, the closer our predictions are to the true labels.

model.fit(train_images, train_labels, epochs=5)
#epachs = how many times the model sees the information

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc", test_acc)

predictions = model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
    #Setting up a way to see the image


