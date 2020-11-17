from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import hydra
import gc

from pathlib import Path

import mlflow
import mlflow.lightgbm


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


mse_scorer = make_scorer(rmse)


def save_log(score, rmse):
    mlflow.log_metrics({
        f"all_rmse": rmse,
        f"all_score - 10000": score
    })
    mlflow.log_artifact(".hydra/config.yaml")
    mlflow.log_artifact(".hydra/hydra.yaml")
    mlflow.log_artifact(".hydra/overrides.yaml")
    mlflow.log_artifact("permutation_importance.log")
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
    target = train["shipping_time"]
    train = train.drop(columns="shipping_time")
    kfold = KFold(n_splits=cfg.base.n_folds, shuffle=True, random_state=cfg.base.seed)
    score = 0

    rand = np.random.randint(0, 1000000)
    experiment_name = f"{'optuna_' if cfg.base.optuna else ''}{rand}"
    print("file:///" + hydra.utils.get_original_cwd() + "mlruns")
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    use_cols = pd.Series(train.columns)
    use_cols.to_csv("features.csv", index=False, header=False)

    imp_df = pd.DataFrame()

    mlflow.lightgbm.autolog()
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):
        x_train, x_valid = train.loc[train_idx], train.loc[valid_idx]
        y_train, y_valid = target[train_idx], target[valid_idx]
        gc.collect()

        clf = lgb.LGBMRegressor(
            num_estimators=1000,

            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            seed=22,
        )
        clf.fit(
            x_train,
            y_train,
            eval_metric="rmse",

            verbose=100,
            early_stopping_rounds=100,
            eval_set=[
                (
                    x_valid,
                    y_valid
                )
            ]
        )

        result = permutation_importance(
            clf,
            x_train,
            y_train,
            scoring=mse_scorer,
            n_repeats=10,
            n_jobs=-1,
            random_state=cfg.base.seed
        )

        perm_imp_df = pd.DataFrame(
            {"importances_mean": result["importances_mean"], "importances_std": result["importances_std"]},
            index=train.columns)

        if fold == 0:
            imp_df = perm_imp_df / cfg.base.n_folds
        else:
            imp_df += perm_imp_df / cfg.base.n_folds

    imp_df.sort_values("importances_mean", ascending=False).to_csv("permutation_importance.csv")
    can_use = imp_df[imp_df.importances_mean < 0]

    can_use.to_csv("can_use.csv")

    print(score)

    mlflow.set_experiment("permutation_importance")
    with mlflow.start_run(run_name=f"{experiment_name}"):

        save_log(score, np.exp(-score) * 10000)
        mlflow.log_artifact("permutation_importance.csv")
        mlflow.log_artifact("can_use.csv")


if __name__ == "__main__":
    main()
