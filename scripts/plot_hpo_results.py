import argparse

import optuna
import plotly.express as px
import pymysql

pymysql.install_as_MySQLdb()


def combine_plotly_figs_to_html(plotly_figs, html_fname, include_plotlyjs="cdn"):
    with open(html_fname, "w") as f:
        f.write(plotly_figs[0].to_html(include_plotlyjs=include_plotlyjs))
        for fig in plotly_figs[1:]:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))


def plot_optuna_default_graphs(optuna_study, params):
    history_plot = optuna.visualization.plot_optimization_history(optuna_study)
    history_by_epoch_plot = optuna.visualization.plot_intermediate_values(optuna_study)
    parallel_plot = optuna.visualization.plot_parallel_coordinate(optuna_study, params=params)
    contour_plot = optuna.visualization.plot_contour(optuna_study, params=params)
    slice_plot = optuna.visualization.plot_slice(optuna_study, params=params)
    param_importance_plot = optuna.visualization.plot_param_importances(optuna_study)
    edf_plot = optuna.visualization.plot_edf(optuna_study)
    plot_list = [history_plot, history_by_epoch_plot, parallel_plot, contour_plot, slice_plot, param_importance_plot,
                 edf_plot]
    return plot_list


def plot_optuna_scatter_custom_values(optuna_study, attributes, y, color=None, size=None, title=None):
    user_attributes_df = optuna_study.trials_dataframe()
    user_attributes_df = user_attributes_df.dropna()
    user_attributes_df[y] = user_attributes_df[y].astype("float32")
    fig = px.scatter(user_attributes_df, x="number", y=y, color=color,
                     size=size, hover_data=attributes, title=title)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna-db", type=str, help="Path to the Optuna Database file",
                        default="mysql://optuna:Optuna#1234@34.168.75.39:3306/OptunaDB")
    parser.add_argument("-n", "--optuna-study-name", type=str, help="Name of the optuna study",
                        default="hyper_search_churn_nn_study3")
    parser.add_argument("-f", "--html-file-path", type=str, help="Path to the HTML Output file",
                        default="outputs/hpo_results.html")
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize",
                                study_name=args.optuna_study_name,
                                storage=args.optuna_db,
                                load_if_exists=True)
    # print best study
    best_trial = study.best_trial
    print(best_trial.params)
    eval_results = best_trial.user_attrs

    print("")
    print("Validation results")
    print("Accuracy \tPrecision \tRecall  \tF1 score   \tROC-AUC")
    print("%s\t%s\t%s\t%s\t%s" % (eval_results["val_accuracy"], eval_results["val_precision"],
                                  eval_results["val_recall"], eval_results["val_f1_score"], eval_results["val_auc"]))
    print("")

    print("Test results")
    print("Accuracy \tPrecision \tRecall  \tF1 score   \tROC-AUC")
    print("%s\t%s\t%s\t%s\t%s" % (eval_results["test_accuracy"], eval_results["test_precision"],
                                  eval_results["test_recall"], eval_results["test_f1_score"], eval_results["test_auc"]))

    # plots = []
    params = ["activation", "dropout", "num_layers", "batch_size", "lr"]
    plots = plot_optuna_default_graphs(study, params)
    plot_attributes = ["number",
                       "user_attrs_val_accuracy",
                       "user_attrs_val_precision",
                       "user_attrs_val_recall",
                       "user_attrs_val_f1_score",
                       "user_attrs_val_auc"]
    f1_plot = plot_optuna_scatter_custom_values(study, plot_attributes, y="user_attrs_val_f1_score",
                                                title="F1 score of the validation set")
    plots.append(f1_plot)

    auc_plot = plot_optuna_scatter_custom_values(study, plot_attributes, y="user_attrs_val_auc",
                                                 title="ROC-AUC of the validation set")
    plots.append(auc_plot)

    combine_plotly_figs_to_html(plotly_figs=plots, html_fname=args.html_file_path)
