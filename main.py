from pathlib import Path
from processing import Processing


STUDENTS_PATH = Path("data/students.csv")
STUDENTS_TEST_PATH = Path("data/students_test.csv")


if __name__ == '__main__':
    processing = Processing()
    processing.loadData(STUDENTS_PATH)
    dataFrame = processing.buildDataFrame(["Growth", "Weight", "Shoe size"])
    X = processing.createX(["Growth", "Weight"])
    Y = processing.createY(["Shoe size"])
    x_train, x_test, y_train, y_test = processing.splitData(X, Y)
    linearRegressionModel = processing.buildLinearRegressionModel(x_train, x_test, y_train, y_test)
    linearRegressionModel.fit(X, Y)
    predict = linearRegressionModel.predict(X)
    dataFrameWithPredict = processing.addPredictColumn("Predicted", predict)
    dataFrameWithPredictRound = processing.roundColumnValues("Predicted")
    processing.showDiff(dataFrameWithPredictRound, "Shoe size", "Predicted", show=True, head="full")
    print(linearRegressionModel)

