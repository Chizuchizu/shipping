from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap
import hydra
import gc

from pathlib import Path

import mlflow
import mlflow.lightgbm


def save_log(score, rmse):
    mlflow.log_metrics({
        f"all_rmse": rmse,
        f"all_score - 10000": score
    })
    mlflow.log_artifact(".hydra/config.yaml")
    mlflow.log_artifact(".hydra/hydra.yaml")
    mlflow.log_artifact(".hydra/overrides.yaml")
    mlflow.log_artifact("train_hydra.log")
    mlflow.log_artifact("features.csv")


@hydra.main(config_name="../config/training.yaml")
def main(cfg):
    cwd = Path(hydra.utils.get_original_cwd())

    if cfg.base.optuna:
        import optuna.integration.lightgbm as lgb
    else:
        import lightgbm as lgb

    data = [pd.read_pickle(cwd / f"../features_data/{f}.pkl") for f in cfg.features]
    data = pd.concat(data, axis=1)

    train = data[data["train"]].drop(columns="train")
    test = data[~data["train"]].drop(columns=["train", "shipping_time"])
    target = train["shipping_time"]
    train = train.drop(columns="shipping_time")

    # if cfg.base.permutation:
    #     imp_cols = pd.read_csv(cwd / f"../features_data/{cfg.permutations}.csv").iloc[:, 0]
    #     train = train[imp_cols]
    #     test = test[imp_cols]

    kfold = KFold(n_splits=cfg.base.n_folds, shuffle=True, random_state=cfg.base.seed)
    pred = np.zeros(test.shape[0])
    score = 0

    rand = np.random.randint(0, 1000000)
    experiment_name = f"{'optuna_' if cfg.base.optuna else ''}{rand}"
    print("file:///" + hydra.utils.get_original_cwd() + "mlruns")
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    use_cols = pd.Series(train.columns)
    use_cols.to_csv("features.csv", index=False, header=False)

    mlflow.lightgbm.autolog()
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):
        x_train, x_valid = train.loc[train_idx], train.loc[valid_idx]
        y_train, y_valid = target[train_idx], target[valid_idx]

        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        # del x_train
        del x_valid
        del y_train
        del y_valid
        gc.collect()
        mlflow.set_experiment(f"fold_{fold + 1}")

        with mlflow.start_run(run_name=f"{experiment_name}"):
            estimator = lgb.train(
                params=dict(cfg.parameters),
                train_set=d_train,
                num_boost_round=cfg.base.num_boost_round,
                valid_sets=[d_train, d_valid],
                verbose_eval=500,
                early_stopping_rounds=100
            )

            y_pred = estimator.predict(test)
            pred += y_pred / cfg.base.n_folds

            print(fold + 1, "done")

            rmse = estimator.best_score["valid_1"]["rmse"]
            score += rmse / cfg.base.n_folds

            """shap"""

            explainer = shap.TreeExplainer(estimator, data=x_train, check_additivity=False)
            tr_x_shap_values = explainer.shap_values(x_train, check_additivity=False)
            _ = shap.summary_plot(shap_values=tr_x_shap_values,
                                  features=x_train,
                                  feature_names=x_train.columns,
                                  show=False)
            plt.savefig("shap_summary.png")
            plt.clf()

            del explainer
            del x_train
            del d_train
            del d_valid

            save_log(rmse, np.exp(-rmse) * 10000)

            mlflow.log_artifact("shap_summary.png")

    print(score)
    # if not DEBUG:
    ss = pd.read_csv(cwd / "../data/submission_2.csv")
    ss["shipping_time"] = pred.round(4)
    ss["shipping_time"].to_csv(cwd / f"../outputs/{rand}.csv", index=False, header=False)

    mlflow.set_experiment("all")
    with mlflow.start_run(run_name=f"{experiment_name}"):

        save_log(score, np.exp(-score) * 10000)
        mlflow.log_artifact(cwd / f"../outputs/{rand}.csv")


if __name__ == "__main__":
    main()
