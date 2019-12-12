import datetime as dt
import itertools as it
import operator as op
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache, partial, reduce, wraps
from glob import glob
from importlib import reload
from itertools import count
from operator import attrgetter as prop
from operator import itemgetter as sel
from os import path
from os.path import *
from pathlib import Path

import altair as A
import dask
import dask.dataframe as dd
import fastparquet
import matplotlib.pyplot as plt
import mystan as ms
# Myutils
import myutils as mu
import numpy as np
import numpy.random as nr
import pandas as pd
import pandas_utils as pu
import pandas_utils3 as p3
import plotnine as p9
import scipy as sp
import seaborn as sns
# !pip install simplejson
import simplejson
import toolz.curried as z
from altair import Chart
from altair import datum as D
from altair import expr as E
from faster_pandas import MyPandas as fp
from fastparquet import ParquetFile
from joblib import Memory
from numba import njit
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal
from plotnine import aes, ggplot, ggtitle, qplot, theme, xlab, ylab
from pyarrow.parquet import ParquetFile as Paf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

ss = lambda x: StandardScaler().fit_transform(x.values[:, None]).ravel()

sns.set_palette("colorblind")
mem = Memory(cachedir="cache", verbose=0)


reload(mu)

ap = mu.dmap

dd.DataFrame.q = lambda self, q, local_dict={}, **kw: self.query(
    q, local_dict=z.merge(local_dict, kw)
)


vc = z.compose(Series, Counter)


def mk_geom(p9, pref="geom_"):
    geoms = [c for c in dir(p9) if c.startswith(pref)]
    geom = lambda: None
    geom.__dict__.update(
        {name[len(pref) :]: getattr(p9, name) for name in geoms}
    )
    return geom


geom = mk_geom(p9, pref="geom_")
facet = mk_geom(p9, pref="facet_")


def run_magics():
    args = [
        "matplotlib inline",
        "autocall 1",
        "load_ext autoreload",
        "autoreload 2",
    ]
    for arg in args:
        get_ipython().magic(arg)


DataFrame.sel = lambda df, f: df[[c for c in df if f(c)]]
Path.g = lambda self, *x: list(self.glob(*x))

pd.options.display.width = 220
pd.options.display.min_rows = 30
A.data_transformers.enable("json", prefix="altair/altair-data")


# def list_wrap(f):
#     @wraps(f)
#     def fn(*a, **k):
#         return

lrange = z.compose(list, range)
lmap = z.compose(list, map)
lfilter = z.compose(list, filter)
