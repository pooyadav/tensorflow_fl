import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)
#num_words = 10000 we take in the most frequent words
#We leave out words that are less frequent as we cant compare those words to other data sets

#print(train_data)
# A dictionary mapping words to an integer index
_word_index = data.get_word_index()
#Gets the prewritten word index

#Adding more words to the dictionary
word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0 #Pad allows us to add words to make each movie review the same length
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

#Values and keys are swapped, so we have the integer to point to a word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#Adding padding to the train data and test data to make the length atleast 250
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])
# this function will return the decoded (human readable) reviews
"""
#This is the same way we made a model in the NeuralNetworks.py just a different syntax
model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16))
#88000 word vectors, each repersents a word, words that have similar meaning have a similar angle
model.add(keras.layers.GlobalAveragePooling1D())
#This layer averages the data
model.add(keras.layers.Dense(16, activation="relu"))
#Dense layer
model.add(keras.layers.Dense(1, activation="sigmoid"))
#Dense layer spits out a sigmoid function, there is not multiple dense layers just 1
#Dense layer looks at word vectors and gives a value between 1 and 0
model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train,y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
#batch size = how many reviews we load in each time
results = model.evaluate(test_data, test_labels)
print(results)

model.save("model.h5")

"""
model = keras.models.load_model("model.h5")

def review_encode(s):
	encoded = [1]
	#starting tag is 1
	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
			#if word is known then append the word
		else:
			encoded.append(2)
			#If the word is unknown append unknown
	return encoded


with open("test.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",maxlen=250)  # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])

