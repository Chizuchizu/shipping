from features.base import Feature, generate_features, create_memo
from src.pre_fun import base_data

import hydra

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


class Groupby_mean(Feature):
    def create_features(self):
        for groupby_col in groupby_cols:
            for calc_col in calc_cols:
                self.data[f"{groupby_col}_{calc_col}"] = data.groupby(groupby_col)[calc_col].transform("mean")
                data[f"{groupby_col}_{calc_col}"] = self.data[f"{groupby_col}_{calc_col}"].copy()

        create_memo(f"gr_mean_{groupby_cols}_{calc_cols}", "groupby_meanです")


class Groupby_diff(Feature):
    def create_features(self):
        for groupby_col in groupby_cols:
            for calc_col in calc_cols:
                self.data[f"{groupby_col}_{calc_col}_diff"] = data[f"{groupby_col}_{calc_col}"] - data[calc_col]
                data[f"{groupby_col}_{calc_col}_diff"] = self.data[f"{groupby_col}_{calc_col}_diff"].copy()

        create_memo(f"gr_diff_{groupby_cols}_{calc_cols}", "groupby_meannoのdiffです")


class Groupby_std(Feature):
    def create_features(self):

        for groupby_col in groupby_cols:
            for calc_col in calc_cols:
                self.data[f"{groupby_col}_{calc_col}_s"] = data.groupby(groupby_col)[calc_col].transform("std")
                data[f"{groupby_col}_{calc_col}_s"] = self.data[f"{groupby_col}_{calc_col}_s"].copy()

        create_memo(f"gr_std_{groupby_cols}_{calc_cols}", "groupby_stdです")


@hydra.main(config_name="../config/features.yaml")
def run(cfg):
    generate_features(globals(), cfg.overwrite)


# デバッグ用
if __name__ == "__main__":
    run()
