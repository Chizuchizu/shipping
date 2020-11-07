from functools import wraps

import time
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

TARGET = "shipping_time"
train = pd.read_csv("../data/train_2_pr.csv").iloc[:, 1:]
test = pd.read_csv("../data/test_2.csv").iloc[:, 1:]
target = train[TARGET]

train = train.drop(columns=TARGET)
"""
pick_upは全てA
source_countryはすべてGB
selectedはすべてY
"""
unnecessary_cols = [
    "pick_up_point",
    "source_country",
    "selected"
]

cat_cols = [
    "shipment_id",
    "drop_off_point",
    "destination_country",
    "shipment_mode",
    "shipping_company"
]


def stop_watch(func_):
    @wraps(func_)
    def wrapper(*args, **kargs):
        # 処理開始直前の時間
        start = time.time()

        # 処理実行
        result = func_(*args, **kargs)

        # 処理終了直後の時間から処理時間を算出
        elapsed_time = time.time() - start

        # 処理時間を出力
        print("{} s in {}".format(round(elapsed_time, 7), func_.__name__))
        return result

    return wrapper


def concat_train_test(train_, test_, drop_cols):
    train_["train"] = True
    test_["train"] = False

    return pd.concat([train_, test_]).drop(columns=drop_cols)


def preprocess(data, cat_cols_):
    for cat_col in cat_cols_:
        data[cat_col] = data[cat_col].astype("category").cat.codes

    memo = "send_timestamp"
    data[memo] = pd.to_datetime(data["send_timestamp"])
    data["year"] = data[memo].apply(lambda x: x.year)
    data["month"] = data[memo].apply(lambda x: x.month)
    data["day"] = data[memo].apply(lambda x: x.day)
    data["weekday"] = data[memo].apply(lambda x: x.dayofweek)
    data["hour"] = data[memo].apply(lambda x: x.hour)

    data = data.drop(columns=memo)
    return data


def feature_engineering(data):
    groupby_cols = ["shipment_mode", "shipping_company", "weekday", "drop_off_point"]
    calc_cols = [TARGET, "freight_cost", "gross_weight", "shipment_charges"]
    data[TARGET] = np.nan
    data.loc[data["train"], TARGET] = target.values

    for groupby_col in groupby_cols:
        for calc_col in calc_cols:
            data[f"{groupby_col}_{calc_col}"] = data.groupby(groupby_col)[calc_col].transform("mean")
            data[f"{groupby_col}_{calc_col}_s"] = data.groupby(groupby_col)[calc_col].transform("std")
            if (calc_col == TARGET) and (groupby_col in ["weekday", "drop_off_point", "shipping_company", "shipment_mode"]):
                continue
            else:
                data[f"{groupby_col}_{calc_col}_diff"] = data[f"{groupby_col}_{calc_col}"] - data[calc_col]

    # data = data.drop(columns=TARGET)

    return data




merged = concat_train_test(train, test, unnecessary_cols)

merged = preprocess(merged, cat_cols)

merged = feature_engineering(merged)

merged.to_pickle("../data/train_test_v2.pkl")
