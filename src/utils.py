import pandas as pd
from sklearn import metrics

import src.constants as const


def load_dataset(path_to_file):
    dataframe = pd.read_csv(path_to_file)
    dataframe.drop(const.DROPPED_COLUMNS, inplace=True)
    return dataframe


class Evaluation:
    def __init__(self, actuals, predictions):
        self.accuracy = metrics.accuracy_score(actuals, predictions)
        self.precision = metrics.precision_score(actuals, predictions)
        self.recall = metrics.recall_score(actuals, predictions)
        self.auc = metrics.roc_auc_score(actuals, predictions)
        self.f1_score = metrics.f1_score(actuals, predictions)
        self.cm = metrics.confusion_matrix(actuals, predictions)

    def print(self):
        print("Accuracy\tPrecision\tRecall\tAUC\tF1")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (self.accuracy, self.precision, self.recall, self.auc, self.f1_score))

        print("Confusion Matrix")
        print(self.cm)
