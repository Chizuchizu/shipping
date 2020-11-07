import lightgbm as lgb
import optuna.integration.lightgbm as lgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

from src.pre_fun import load_data

import mlflow
import mlflow.lightgbm


N_FOLDS = 4
VERSION = 2
DEBUG = True
NUM_CLASSES = 4
SEED = 22
num_rounds = 2000

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    # "feature_fraction": 0.7,
    # 'max_depth': 7,
    # 'num_leaves': 64,
    # 'max_bin': 31,
    # 'nthread': -1,
    # 'bagging_freq': 1,
    'verbose': -1,
    'seed': SEED,
}

# data = pd.read_pickle(f"../data/train_test_v{VERSION}.pkl")
data = load_data()
train = data[data["train"]].drop(columns="train")
test = data[~data["train"]].drop(columns=["train", "shipping_time"])
target = train["shipping_time"]
train = train.drop(columns="shipping_time")
kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
pred = np.zeros(test.shape[0])
score = 0

rand = np.random.randint(0, 1000000)

mlflow.lightgbm.autolog()
for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):
    x_train, x_valid = train.loc[train_idx], train.loc[valid_idx]
    y_train, y_valid = target[train_idx], target[valid_idx]

    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    del x_train
    del x_valid
    del y_train
    del y_valid
    gc.collect()
    mlflow.set_experiment(f'training_{rand}')

    with mlflow.start_run(run_name=f"fold_{fold+1}"):
        estimator = lgb.train(
            params=params,
            train_set=d_train,
            num_boost_round=num_rounds,
            valid_sets=[d_train, d_valid],
            verbose_eval=100,
            early_stopping_rounds=100
        )

        y_pred = estimator.predict(test)
        pred += y_pred / N_FOLDS

        print(fold + 1, "done")

        score += estimator.best_score["valid_1"]["rmse"] / N_FOLDS

        mlflow.log_metrics({
            f"fold_{fold+1}_rmse": estimator.best_score["valid_1"]["rmse"]
        })

mlflow.set_experiment(f'training_{rand}')
with mlflow.start_run(run_name="all"):

    mlflow.log_metrics({
        f"all_rmse": score,
        f"all_rmse_round": round(score, 4)
    })

    mlflow.log_param("columns", train.columns)


print(score)
# if not DEBUG:
ss = pd.read_csv("../data/submission_2.csv")
ss["shipping_time"] = pred.round(4)
ss["shipping_time"].to_csv(f"../outputs/lgbm_{rand}.csv", index=False, header=False)
