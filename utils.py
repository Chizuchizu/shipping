"""
https://github.com/KazukiOnodera/Instacart/blob/416b6b0220d3aed62c8d323caa3ee46f4b614a72/py_feature/utils.py
"""

import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
# import pickle
from time import time
from datetime import datetime
import gc

# from itertools import chain
warnings.filterwarnings("ignore")


# =============================================================================
# def
# =============================================================================
def start(fname):
    global st_time
    st_time = time()
    print("""
#==============================================================================
# START!!! {}    PID: {}    time: {}
#==============================================================================
""".format(fname, os.getpid(), datetime.today()))

    #    send_line(f'START {fname}  time: {elapsed_minute():.2f}min')

    return


def end(fname):
    print("""
#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(fname))
    print('time: {:.2f}min'.format(elapsed_minute()))

    #    send_line(f'FINISH {fname}  time: {elapsed_minute():.2f}min')

    return


@contextmanager
def timer(name):
    t0 = time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time() - t0:.0f} s')


def elapsed_minute():
    return (time() - st_time) / 60


def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)


