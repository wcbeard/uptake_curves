# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook ("") tries to download all recent experiments via dbx

# %%
from boot_utes import (reload, add_path, path, run_magics)
add_path('..', '../src/', '~/repos/myutils/', )

from utils.uptake_curves_imps import *; exec(pu.DFCols_str); exec(pu.qexpr_str); run_magics()
# import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

sns.set_style('whitegrid')
A.data_transformers.enable('json', prefix='data/altair-data')
S = Series; D = DataFrame

from big_query import bq_read
from pyspark_utils import C, spark, sql

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# %%

# %% [markdown]
# # Load
