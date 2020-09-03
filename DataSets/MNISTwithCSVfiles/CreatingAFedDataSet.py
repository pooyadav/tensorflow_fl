import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
#pandas used to import csv files
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")