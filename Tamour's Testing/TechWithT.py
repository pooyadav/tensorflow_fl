import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";") #Seperate using semi colons
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print (data.head())

predict = "G3" #This is a label this is what you are trying to predict

X = np.array(data.drop([predict], 1)) #Our Data without G3
y = np.array(data[predict]) #These are all the g3 values
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

best = 0
"""
for _ in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
    #What you are doing here is splitting up X into a training set and a test set same with the Y
    #This allows you to test x against Y, the algorithm uses 90% of the training data to generate a model
    #90% of training data is randomly selected

    linear = linear_model.LinearRegression() #Creating a linear model

    linear.fit(x_train, y_train) #Training the data together to make a linear line
    acc = linear.score(x_test, y_test) #Testing the accuracy of our linear algorithm
    #You cannot check the accuracy without fitting the lines together
    print(acc)

    #Saving the model
    if acc > best:
        best = acc
        with open ("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
    
"""


#Opening the model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: \n", + linear.coef_)
print ("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len(predictions)):
    print (predictions[x], x_test[x], y_test[x])

p = "studytime"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

