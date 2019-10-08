import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

digits = load_digits();
logReg = LogisticRegression()


xTrain, xTest, yTrain, yTest = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 0)

logReg.fit(xTrain, yTrain)

c = logReg.predict(xTest)

print(c)