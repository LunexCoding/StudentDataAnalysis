import pandas as pd
from sklearn.model_selection import train_test_split
from linearRegression import LinearRegressionModel


class Processing:
    def __init__(self):
        self.__data = None

    def loadData(self, path):
        self.__loadData(path)
        self.__preprocessingData()

    def __loadData(self, path):
        self.__data = pd.read_csv(path)

    def __preprocessingData(self):
        self.__data = self.__data.fillna(0)

    def buildDataFrame(self, columns):
        return self.__buildDataFrame(columns)

    def __buildDataFrame(self, columns):
        return self.__data[columns]

    def splitData(self, X, Y, randomState=42, testSize=0.2):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=randomState, test_size=testSize)
        return x_train, x_test, y_train, y_test

    def buildLinearRegressionModel(self, x_train, x_test, y_train, y_test):
        model = LinearRegressionModel()
        model.loadData(x_train, x_test, y_train, y_test)
        return model

    def clearData(self):
        self.__clearData()

    def __clearData(self):
        self.__data = None




