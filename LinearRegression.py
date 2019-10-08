import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./Datasets/salaryData.csv")


x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=1 / 3, random_state=0)

linearRegressor = LinearRegression()

linearRegressor.fit(xTrain, yTrain)

# xT = np.array([[3.3433]])

yPrediction = linearRegressor.predict(xTest)
# print(yPrediction)

# plot.scatter(xTest, yTest, color='red')
# plot.plot(xTrain, linearRegressor.predict(xTrain), color='blue')
# plot.title('Salary vs Experience (Test set)')
# plot.xlabel('Years of Experience')
# plot.ylabel('Salary')
# plot.show()
