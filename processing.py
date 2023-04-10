import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from linearRegression import LinearRegressionModel


class Processing:
    def __init__(self):
        self.__data = None
        self.__tableFormat = "github"

    # loading data
    def loadData(self, path):
        self.__loadData(path)
        self.__preprocessingData()

    def __loadData(self, path):
        self.__data = pd.read_csv(path)

    def __preprocessingData(self):
        self.__data = self.__data.fillna(0)

    # build DataFrame
    def buildDataFrame(self, columns):
        assert isinstance(columns, list)
        self.__buildDataFrame(columns)
        return self.__getDataFrame()

    def __buildDataFrame(self, columns):
        self.__data = self.__data[columns]

    def __getDataFrame(self):
        return self.__data

    # split data by X, Y
    def createX(self, columns):
        assert isinstance(columns, list)
        return self.__createX(columns)

    def __createX(self, columns):
        return self.__data[columns]

    def createY(self, columns):
        assert isinstance(columns, list)
        return self.__createY(columns)

    def __createY(self, columns):
        return self.__data[columns]

    # create train and test samplings
    def splitData(self, X, Y, randomState=42, testSize=0.2):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=randomState, test_size=testSize)
        return x_train, x_test, y_train, y_test

    def buildLinearRegressionModel(self, x_train, x_test, y_train, y_test):
        model = LinearRegressionModel()
        model.loadData(x_train, x_test, y_train, y_test)
        return model

    def addPredictColumn(self, columnName, data):
        self.__data[columnName] = data
        return self.__getDataFrame()

    # operations with DataFrame
    def roundColumnValues(self, columnName, digits=0):
        self.__roundColumnValues(columnName, digits)
        return self.__getDataFrame()

    def __roundColumnValues(self, columnName, digits):
        self.__data[columnName] = self.__data[columnName].apply(lambda x: round(x, digits))

    def showDiff(self, dataFrame, Y, predicted, show=False, head=20):
        self.__showDiff(dataFrame, Y, predicted, show, head)

    def __showDiff(self, dataFrame, nameColumn_Y, predicted, show=False, head=20):
        dataFrame["diff"] = None
        dataFrame["diff"] = dataFrame.apply(lambda x: self.__checkDiff(x[nameColumn_Y], x[predicted]), axis=1)
        if show:
            if isinstance(head, int):
                self.displayDataFrame(dataFrame, head)
                print(self.__countDiff(dataFrame, head))
            elif head == "full":
                self.displayDataFrame(dataFrame, head="full")
                print(self.__countDiff(dataFrame, len(dataFrame)))

    def __checkDiff(self, nameColumn_Y, predicted):
        return False if predicted == nameColumn_Y else True

    def __countDiff(self, dataFrame, head):
        return dataFrame["diff"][:head].value_counts()

    def displayDataFrame(self, dataFrame, head=0):
        if isinstance(head, int):
            print(tabulate(dataFrame[:head], headers="keys", tablefmt=self.__tableFormat))
        elif head == "full":
            print(tabulate(dataFrame, headers="keys", tablefmt=self.__tableFormat))

    @property
    def tableFormat(self):
        return self.__tableFormat

    @tableFormat.setter
    def tableFormat(self, tableFormat):
        assert isinstance(tableFormat, str)
        self.__tableFormat = tableFormat

    def __clearData(self):
        self.__data = None

    def __del__(self):
        self.__clearData()



