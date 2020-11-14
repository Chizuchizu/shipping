from features.base import Feature, generate_features
from src.pre_fun import base_data

Feature.dir = "../features_data"
Feature.data = base_data()


class Cost(Feature):
    def create_features(self):
        self.data["cost"] = self.data["gross_weight"] * self.data["freight_cost"]


# class Load(Feature):
#     def load(self):
#         self.data = base_data()


if __name__ == "__main__":
    # data = base_data()

    generate_features(globals())
