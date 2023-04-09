from sklearn.linear_model import LinearRegression as LinearRegression


class LinearRegressionModel:
    def __init__(self):
        self.__clf = LinearRegression()
        self.__x_train = None
        self.__x_test = None
        self.__y_train = None
        self.__y_test = None
        self.__scores = {}

    def loadData(self, x_train, x_test, y_train, y_test):
        self.__loadData(x_train, x_test, y_train, y_test)

    def __loadData(self, x_train, x_test, y_train, y_test):
        self.__x_train = x_train
        self.__x_test = x_test
        self.__y_train = y_train
        self.__y_test = y_test

    def fit(self, x_train, y_train):
        self.__clf.fit(x_train, y_train)

    def predict(self, X):
        self.__calcScores()
        return self.__clf.predict(X)

    def __calcScores(self):
        self.__scores["score"] = self.__clf.score(self.__x_test, self.__y_test)
        self.__scores["coef"] = self.__clf.coef_
        self.__scores["intercept"] = self.__clf.intercept_

    def __str__(self):
        return f"score: {self.__scores['score']}\n" \
               f"coef: {self.__scores['coef']}\n" \
               f"intercept: {self.__scores['intercept']}\n"
