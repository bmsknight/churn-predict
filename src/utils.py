import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder

import src.constants as const


def load_dataset(path_to_file):
    dataframe = pd.read_csv(path_to_file)
    dataframe = transform_columns(dataframe)
    X = dataframe.drop(const.TARGET_COLUMN, axis=1)
    y = dataframe[const.TARGET_COLUMN]
    return X, y


def transform_columns(dataframe):
    encoder = LabelEncoder()
    for col in const.BINARY_COLUMNS:
        dataframe[col] = dataframe[col].map(const.BINARY_MAPPING)
    for col in const.CATEGORICAL_COLUMNS:
        dataframe[col] = encoder.fit_transform(dataframe[col])
    return dataframe


class Standardizer:

    def __init__(self, columns_to_standardize):
        self.scaler = StandardScaler()
        self.columns = columns_to_standardize

    def fit(self, dataframe):
        temp_df = dataframe[self.columns]
        self.scaler.fit(temp_df)

    def transform(self, dataframe):
        temp_df = dataframe.copy(deep=True)
        transformed_data = self.scaler.transform(temp_df[self.columns])
        temp_df[self.columns] = transformed_data
        return temp_df

    def fit_transform(self, dataframe):
        self.fit(dataframe)
        return self.transform(dataframe)

    def inverse_transform(self, dataframe):
        temp_df = dataframe.copy(deep=True)
        transformed_data = self.scaler.inverse_transform(temp_df[self.columns])
        temp_df[self.columns] = transformed_data
        return temp_df


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
        print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (self.accuracy, self.precision, self.recall, self.auc, self.f1_score))

        print("Confusion Matrix")
        print(self.cm)


def store_hpo_eval_results(trial, eval_results, prefix=""):
    attributes = eval_results.__dict__
    for key in attributes:
        trial.set_user_attr(prefix + key, str(attributes[key]))


def plot_history(history, trial_id, output_path):
    epochs, losses = zip(*history.items())
    train_loss, val_loss = zip(*[(loss["train_loss"], loss["val_loss"]) for loss in losses])
    plt.style.use("ggplot")
    fig = plt.figure(figsize=[16, 10])
    title = fig.suptitle(f"Train and Val Loss plot for Trial {trial_id}")
    title.set_fontsize(20)
    ax = fig.add_subplot()
    x_axis = ax.xaxis
    y_axis = ax.yaxis
    x_axis.set_label_text("Epoch")
    y_axis.set_label_text("Loss (BCE)")

    train_plot, = ax.plot(epochs, train_loss)
    train_plot.set_linewidth(5)
    test_plot, = ax.plot(epochs, val_loss)
    test_plot.set_linewidth(5)

    fig.legend(handles=[train_plot, test_plot], labels=["Train loss", "Val loss"])
    fig.tight_layout()
    fig.savefig(output_path + "_loss.png")
