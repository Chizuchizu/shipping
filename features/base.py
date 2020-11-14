import argparse
import inspect
import csv
import re
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd

from utils import timer


def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in ({k: v for k, v in namespace.items()}).items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace):
    for f in get_features(namespace):
        if f.data_path.exists():
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')

        # ユーザーに登録してもらう
        # self.data = pd.read_pickle("../features_data/data.pkl")
        self.data_path = Path(self.dir) / f"{self.name}.pkl"

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''

            self.data.columns = prefix + self.data.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.data.to_pickle(str(self.data_path))

    def load(self):

        self.data = pd.read_pickle(str(self.data_path))


def create_memo(col_name, desc):
    file_path = Feature.dir + "/_features_memo.csv"

    if not os.path.isfile(file_path):
        with open(file_path, "w"): pass

    with open(file_path, "r+") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        col = [line for line in lines if line.split(",")[0] == col_name]
        if len(col) != 0: return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])
