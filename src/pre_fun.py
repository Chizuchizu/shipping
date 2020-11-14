from functools import wraps
from fbprophet.make_holidays import make_holidays_df

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
    year_list = [2019, 2020]
    holidays = make_holidays_df(year_list=year_list, country='UK')
    hol = pd.DataFrame(holidays["ds"])  # .rename(columns={"ds": "send_timestamp"})
    hol["is_hol"] = 1

    for cat_col in cat_cols_:
        data[cat_col] = data[cat_col].astype("category").cat.codes

    memo = "send_timestamp"
    data[memo] = pd.to_datetime(data["send_timestamp"])
    data["year"] = data[memo].apply(lambda x: x.year)
    data["month"] = data[memo].apply(lambda x: x.month)
    data["day"] = data[memo].apply(lambda x: x.day)
    data["weekday"] = data[memo].apply(lambda x: x.dayofweek)
    data["hour"] = data[memo].apply(lambda x: x.hour)

    data["ds"] = pd.to_datetime(data[memo].dt.date)

    # data["is_hol"] = data["ds"].apply(lambda x: x in hol["ds"].to_list())
    # data["is_hol"] = data["is_hol"].fillna(0)

    data = data.drop(columns=[memo, "ds"])
    return data


def feature_engineering(data):
    data["cost"] = data["gross_weight"] * data["freight_cost"]

    # data["month_count"] = data.groupby(["year", "month"])["cost"].transform("count")
    # data["day_count"] = data.groupby(["year", "month", "day"])["cost"].transform("count")
    # data["weekday_count"] = data.groupby(["year", "month", "weekday"])["cost"].transform("count")
    # count_list = ["month", "day", "weekday", "hour"]
    # for count_col in count_list:
    #     data[f"{count_col}_diff"] = data[f"{count_col}_count"] - data.groupby(count_col)["cost"].transform("count")

    groupby_cols = ["shipment_mode", "weekday", "drop_off_point", "shipping_company", "hour", "destination_country",
                    "shipment_id"]
    # calc_cols = [TARGET, "freight_cost", "gross_weight", "shipment_charges", "cost", "month_count", "day_count", "weekday_count"]
    calc_cols = ["freight_cost", "gross_weight", "shipment_charges", "cost"]
    data[TARGET] = np.nan
    data.loc[data["train"], TARGET] = target.values

    for groupby_col in groupby_cols:
        for calc_col in calc_cols:
            if (groupby_col == "shipment_id") and (calc_col == TARGET):
                continue
            else:
                data[f"{groupby_col}_{calc_col}"] = data.groupby(groupby_col)[calc_col].transform("mean")
                data[f"{groupby_col}_{calc_col}_diff"] = data[f"{groupby_col}_{calc_col}"] - data[calc_col]

                # data[f"{groupby_col}_{calc_col}_s"] = data.groupby(groupby_col)[calc_col].transform("std")
                # if (calc_col == TARGET) and (groupby_col in ["shipping_mode"]):
                #     data[f"{groupby_col}_{calc_col}_diff"] = data[f"{groupby_col}_{calc_col}"] - data[calc_col]
                # if not calc_col == TARGET:
                #     data[f"{groupby_col}_{calc_col}_diff"] = data[f"{groupby_col}_{calc_col}"] - data[calc_col]

    # data = data.drop(columns=TARGET)

    """
    target encoding
    """

    # data["te_month_mode"] = data.groupby(["year", "month", "shipment_mode"])[TARGET].transform("mean")
    # data["te_month_mode_country"] = data.groupby(["year", "month", "shipment_mode", "destination_country"])[
    #     TARGET].transform("mean")
    # data["te_hour_mode"] = data.groupby(["hour", "shipment_mode"])[TARGET].transform("mean")
    return data


def load_data():
    merged = concat_train_test(train, test, unnecessary_cols)

    merged = preprocess(merged, cat_cols)

    merged = feature_engineering(merged)
    return merged


def base_data():
    return preprocess(concat_train_test(train, test, unnecessary_cols), cat_cols)


print()