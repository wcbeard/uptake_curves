# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook ("") tries to download all recent experiments via dbx
#
# - [ ] altair: checkbox selection

# %%
from boot_utes import reload, add_path, path, run_magics

add_path("..", "../uptake/", "~/repos/myutils/")

from utils.uptake_curves_imps import *

exec(pu.DFCols_str)
exec(pu.qexpr_str)

run_magics()
# import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

sns.set_style("whitegrid")
A.data_transformers.enable("json", prefix="data/altair-data")
S = Series
D = DataFrame

from big_query import bq_read

import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# %%
import altair.vegalite.v3 as alt
import pandas as pd
import numpy as np

rand = np.random.RandomState(42)

df = pd.DataFrame({"xval": range(100), "yval": rand.randn(100).cumsum()})

slider1 = alt.binding_range(min=0, max=100, step=1, name="cutoff1:")
selector1 = alt.selection_single(
    name="SelectorName1", fields=["cutoff1"], bind=slider1, init={"cutoff1": 50}
)

slider2 = alt.binding_range(min=0, max=100, step=1, name="cutoff2:")
selector2 = alt.selection_single(
    name="SelectorName2", fields=["cutoff2"], bind=slider2, init={"cutoff2": 50}
)

ch_base = (
    alt.Chart(df)
    .mark_point()
    .encode(
        x="xval",
        y="yval",
        color=alt.condition(
            alt.datum.xval < selector1.cutoff1, alt.value("red"), alt.value("blue")
        ),
    )
)

ch1 = ch_base.add_selection(selector1)

ch2 = ch_base.encode(
    color=alt.condition(
        alt.datum.xval < selector2.cutoff2, alt.value("red"), alt.value("blue")
    )
).add_selection(selector2)


ch1 & ch2
