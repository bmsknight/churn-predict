import argparse
import random
import time

import optuna
import pymysql
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

import src.constants as const
from src.nn_hpo_utils import TunableFeedForwardNN
from src.nn_utils import ChurnDataset, Learner
from src.utils import load_dataset, Evaluation, store_hpo_eval_results, plot_history, Standardizer

# Fix for optuna accessing MySQLdb module which is not present
# alternative packages are not present in compute canada
# Hence need to use pymysql as below
pymysql.install_as_MySQLdb()


def main(config, trial_number):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print("Training will happen on : ", device)

    X, y = load_dataset(const.DATASET_PATH)

    # This will result in 0.6 train, 0.2 validation, 0.2 test splits
    # The random state is set to a fixed value to ensure that the splits stay same across the runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True, random_state=0)

    scaler = Standardizer(columns_to_standardize=const.NUMERIC_COLUMNS+const.CATEGORICAL_COLUMNS)
    # scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = ChurnDataset(X_train, y_train)
    val_dataset = ChurnDataset(X_val, y_val)
    test_dataset = ChurnDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = TunableFeedForwardNN(in_features=X.shape[1], num_layers=config["num_layers"],
                                 num_neurons=config["num_neurons"], activation=config["activation"],
                                 drop_out=config["dropout"], batch_normalization=True)
    model = model.to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    model_trainer = Learner(model, loss_fn, optimizer, const.N_EPOCHS, device,
                            model_save_path=const.TEMPORARY_MODEL_SAVE_PATH + str(trial_number.number))
    history = model_trainer.train(train_loader, val_loader)
    model_trainer.load_best_model()
    y_val_pred = model_trainer.predict(val_loader)
    y_test_pred = model_trainer.predict(test_loader)

    print("Validation Results")
    val_results = Evaluation(actuals=y_val, predictions=y_val_pred)
    val_results.print()
    store_hpo_eval_results(trial_number, val_results, prefix="val_")
    print("Test Results")
    test_results = Evaluation(actuals=y_test, predictions=y_test_pred)
    test_results.print()
    store_hpo_eval_results(trial_number, test_results, prefix="test_")

    # The model selection will be done based on the returned loss. We are returning the validation AUC for that
    plot_history(history=history, trial_id=trial_number.number,
                 output_path=const.OUTPUT_SAVE_FORMAT.format(run_id=trial_number.number))
    return val_results.auc


def objective(trial):
    params = dict()
    params["lr"] = trial.suggest_loguniform("lr", 1e-6, 5e-1)
    params["batch_size"] = trial.suggest_int("batch_size", 32, 2048, log=True)
    # There is an already mandatory last layer
    params["num_layers"] = trial.suggest_int("num_layers", 1, 9)
    params["num_neurons"] = []
    for l in range(params["num_layers"]):
        neurons = trial.suggest_categorical("n_units_l{}".format(l), [4, 8, 16, 32, 64, 256])
        params["num_neurons"].append(neurons)
    params["activation"] = trial.suggest_categorical("activation",
                                                     ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "selu"])
    params["dropout"] = trial.suggest_float("dropout", 0, 0.5, step=0.1)

    print(f"Initiating Run {trial.number} with params : {trial.params}")

    loss = main(params, trial)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna-db", type=str, help="Path to the Optuna Database file",
                        default="mysql://optuna:Optuna#1234@34.168.75.39:3306/OptunaDB")
    parser.add_argument("-n", "--optuna-study-name", type=str, help="Name of the optuna study",
                        default="hyper_search_churn_nn_study1")
    args = parser.parse_args()

    # wait for some time to avoid overlapping run ids when running parallel
    wait_time = random.randint(0, 10) * 3
    print(f"Waiting for {wait_time} seconds before starting")
    time.sleep(wait_time)

    study = optuna.create_study(direction="maximize",
                                study_name=args.optuna_study_name,
                                storage=args.optuna_db,
                                load_if_exists=True,
                                )
    study.optimize(objective, n_trials=1)
