from features.base import Feature, generate_features, create_memo
from src.pre_fun import base_data

import cudf
import hydra
import numpy as np
from sklearn.model_selection import KFold
from xfeat import Pipeline, SelectNumerical, ArithmeticCombinations, TargetEncoder

Feature.dir = "../features_data"
data = base_data()

groupby_cols = ["shipment_mode", "weekday", "drop_off_point", "shipping_company", "hour", "destination_country",
                "shipment_id"]
calc_cols = ["freight_cost", "gross_weight", "shipment_charges", "cost"]


class Cost(Feature):
    def create_features(self):
        self.data["cost"] = data["gross_weight"] * data["freight_cost"]
        data["cost"] = self.data["cost"].copy()
        create_memo("cost", "実際にかかった費用")


class Base_data(Feature):
    def create_features(self):
        self.data = data.drop(columns=["processing_days", "cut_off_time"])
        create_memo("base_data", "初期")


class Xfeat_data(Feature):
    def create_features(self):
        self.data = data.drop(columns=["train"])

        self.data = Pipeline(
            [
                SelectNumerical(),
                ArithmeticCombinations(
                    exclude_cols=["shipping_time", "train"], drop_origin=True, operator="+", r=2,
                ),
            ]
        ).fit_transform(data).reset_index(
            drop=True
        )

        self.data["train"] = data["train"].copy()
        self.data["shipping_time"] = data["shipping_time"].copy()

        create_memo("xfeat_data", "xfeatで作った特徴量")


class Target_Encoding(Feature):
    def create_features(self):
        # ごめんなさい　関数やるの面倒だった
        fold = KFold(n_splits=4, shuffle=True, random_state=22)
        encoder = TargetEncoder(
            input_cols=[
                "shipment_mode", "weekday", "drop_off_point", "shipping_company", "hour", "destination_country"
            ],
            target_col="shipping_time",
            # output_prefix="te_",
            fold=fold
        )

        self.data = encoder.fit_transform(cudf.from_pandas(data.fillna(0)))

        use_cols = [col for col in self.data.columns if "_te" in col]
        self.data = self.data[use_cols].to_pandas()

        create_memo("target_encoding", "xfeatでやった")


class Processing_days(Feature):
    def create_features(self):
        self.data["processing_days"] = (data["processing_days"] == "24/7").astype(int)
        create_memo("processing_days", "年中無休で港が開いているならTrue")


class Cut_off_time(Feature):
    def create_features(self):
        self.data["cut_off_time"] = (data["cut_off_time"] == "24/7").astype(int)
        create_memo("cut_off_time", "いつでも商品を受け取ることができるならTrue")


class Groupby_mean(Feature):
    def create_features(self):
        for groupby_col in groupby_cols:
            for calc_col in calc_cols:
                self.data[f"{groupby_col}_{calc_col}"] = data.groupby(groupby_col)[calc_col].transform("mean")
                data[f"{groupby_col}_{calc_col}"] = self.data[f"{groupby_col}_{calc_col}"].copy()

        create_memo("groupby_mean", f"gr_mean_{groupby_cols}_{calc_cols}")


class Groupby_diff(Feature):
    def create_features(self):
        for groupby_col in groupby_cols:
            for calc_col in calc_cols:
                self.data[f"{groupby_col}_{calc_col}_diff"] = data[f"{groupby_col}_{calc_col}"] - data[calc_col]
                data[f"{groupby_col}_{calc_col}_diff"] = self.data[f"{groupby_col}_{calc_col}_diff"].copy()

        create_memo("groupby_diff", f"gr_diff_{groupby_cols}_{calc_cols}")


class Groupby_std(Feature):
    def create_features(self):

        for groupby_col in groupby_cols:
            for calc_col in calc_cols:
                self.data[f"{groupby_col}_{calc_col}_s"] = data.groupby(groupby_col)[calc_col].transform("std")
                data[f"{groupby_col}_{calc_col}_s"] = self.data[f"{groupby_col}_{calc_col}_s"].copy()

        create_memo("groupby_std", f"gr_std_{groupby_cols}_{calc_cols}")


# https://github.com/tks0123456789/kaggle-Walmart_Trip_Type
def sign_log1p_abs(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Sign_log1p_abs(Feature):
    def create_features(self):
        for calc_col in calc_cols:
            self.data[f"sign_log1p_abs_{calc_col}"] = data[calc_col].apply(sign_log1p_abs)

        create_memo(f"sign_log1p_abs_{calc_cols}", "tkmさんの特徴量、それっぽいやつ")


@hydra.main(config_name="../config/features.yaml")
def run(cfg):
    generate_features(globals(), cfg.overwrite)


# デバッグ用
if __name__ == "__main__":
    run()
