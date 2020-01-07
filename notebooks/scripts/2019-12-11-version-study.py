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

# %%
from boot_utes import reload, add_path, path, run_magics

add_path("..", "../src/", "~/repos/myutils/")

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
# # add_path("/Users/wbeard/repos/missioncontrol-v2/mc2/")
# # add_path("/Users/wbeard/repos/missioncontrol-v2/mc2/data")
# # import data.release_versions as rv

# add_path("/Users/wbeard/repos/dscontrib-moz/src/dscontrib")
# # import wbeard.buildhub_bid as bh
# from requests import get

# %%
import src.data.buildhub_utils as bh
import src.utils.uptake_utes as uu
import src.data.release_dates as rv
import altair.vegalite.v3 as a3

def proc(df):
    df = df.assign(mvers=lambda x: x.dvers.map(uu.maj_os))
    return df


# %%
# Uptake
dfc = bq_read("select * from analysis.wbeard_uptake_vers")
print(len(dfc))

# %%
dfcn = dfc.query("chan == 'nightly'")
# bq_read("select * from analysis.wbeard_uptake_vers where chan = 'nightly'")
dfcn = proc(dfcn)
len(dfcn)

# %% [markdown]
# ## Product details: get current version

# %%
prod_details = rv.read_product_details_all()

# %%
pd_rls = prod_details.query("category == 'major' & date > '2019'")
# pd_beta = prod_details.query("category == 'dev' & date > '2019'")

# %% [markdown]
# ## Beta release

# %%
Chart


# %%
# import data.queries as qs

@mem.cache
def bq_read_cache(*a, **k):
    return bq_read(*a, **k)


def sub_days_null(s1, s2, fillna=None):
    diff = s1 - s2
    not_null = diff.notnull()
    nn_days = diff[not_null].astype("timedelta64[D]").astype(int)
    diff.loc[not_null] = nn_days
    return diff


# %%
_bh_rls_dates = buildhub_data_beta.rename(
    columns={"pub_date": "date", "disp_vers": "version"}
)[["date", "version"]].query(f"date >= '{dfcb.submission_date.min()}'")

# bh_beta_rc = get_beta_release_dates()
beta_release_dates = get_beta_release_dates()

# %%
beta_release_dates = rv.get_beta_release_dates(min_build_date='2019', min_pd_date='2019-01-01')
rv.latest_n_release_beta(beta_release_dates, '2019-12-02', n_releases=3)


# %%
# pd_beta = prod_details.query("category == 'dev' & date > '2019'")
# pd_beta = pull_bh_data_beta()

# %% [markdown]
# ## Release

# %%
# import src.data.release_dates as rv

# def maj_vers(vers: str) -> int:
#     majv, *_ = vers.split(".")
#     return int(majv)

# buildhub_data_rls = rd.pull_bh_data_rls("2019")
# buildhub_data_beta = rd.pull_bh_data_beta('2019')
# bh_data = pd.concat([buildhub_data_rls, buildhub_data_beta], ignore_index=True)

# %%
def check_1to1_mapping(s1, s2):
    dmap = dict(zip(s1, s2))
    s2_should_be = s1.map(dmap)
    ne = s2_should_be != s2
    
    assert ne.sum() == 0, 'Not 1-1 mapping!'
    
# check_1to1_mapping(bh_data.build_id, bh_data.disp_vers)
# check_1to1_mapping(bh_data.build_id, bh_data.pub_date)


# %%
buildhub_data_rls.query("is_major").drop_duplicates(subset='disp_vers', keep='last')

# %% [markdown]
# # Compare to actual uptake
# ## Release

# %% [markdown]
# We have
# * uptake data: `dfc`
# * product details
#     * major: `maj_json` <- https://product-details.mozilla.org/1.0/firefox_history_major_releases.json
#     * all: `pd_rls` <- https://product-details.mozilla.org/1.0/firefox.json
# * buildhub: `buildhub_data_rls`

# %%

# %%
dfc[:3]

# %%
dfcr = dfc.query("chan == 'release'")
# .query("submission_date >= '2019-12-01'")

# %%
w = dfcr.query("os == 'Windows_NT'")

# %%
win_rls = (
    w.groupby(["submission_date", "vers"])
    .n.sum()
    .reset_index(drop=0)
    .assign(dayn=lambda x: x.groupby("submission_date").n.transform("sum"))
    .assign(n_perc=lambda x: x.n / x.dayn * 100)
    .query("n_perc > 1")
)

# %%
maj_json[:3]

# %%
import src.uptake_plots as up

up.mk_rls_plot('')

# %% {"jupyter": {"outputs_hidden": true}}
os = 'Windows_NT'
# os = "Darwin"
# os = "Linux"
charts = [
    mk_rls_plot(date, dfcr, v, os=os)[1]
    for date, v in pd_rls[["date", "version"]]
    .query(f"date >= '{dfcr.submission_date.min()}'")
    .itertuples(index=False)
]

charts = A.vconcat(*charts)
with A.data_transformers.enable("default"):
    charts.save(f"../reports/figures/prod_det_uptake_{os}.png")
charts

# %% {"jupyter": {"outputs_hidden": true}}
charts = [mk_rls_plot(date, dfcr, v)[1]
 for date, v in maj_json.query(f"date >= '{dfcr.submission_date.min()}'").itertuples(index=False)]

charts = A.vconcat(*charts)
with A.data_transformers.enable('default'):
    charts.save('../reports/figures/maj_prod_det_uptake.png')
charts

# %% [markdown]
# ## Beta

# %%
dfcb = dfc.query("chan == 'beta'")

# %%
# w2, hh = mk_rls_plot("2019-12-03", dfcr)

# %% [markdown]
# ### Prod-details dates

# %%
os = "Windows_NT"
# os = "Darwin"
# os = "Linux"

_pd_rls_dates = pd_rls[["date", "version"]].query(
    f"date >= '{dfcb.submission_date.min()}'"
)
charts = [
    mk_rls_plot(date, dfcb, v, os=os, vers_col="dvers", min_max_pct=3)[1]
    for date, v in _pd_rls_dates.itertuples(index=False)
]

charts = A.vconcat(*charts)
# with A.data_transformers.enable("default"):
#     charts.save(f"../reports/figures/prod_det_uptake_{os}.png")
charts

# %% [markdown]
# ### Buildhub release dates

# %%
_bh_rls_dates = buildhub_data_beta.rename(
    columns={"pub_date": "date", "disp_vers": "version"}
)[["date", "version"]].query(f"date >= '{dfcb.submission_date.min()}'")

# %%


_bh_rls_dates.assign(rc=lambda x: x.version.map(is_rc)).query("rc")

# %%

# %%
pd_beta

# %%
os = "Windows_NT"
# os = "Darwin"
# os = "Linux"


charts = [
    mk_rls_plot(date, dfcb, v, os=os, vers_col="dvers", min_max_pct=3)[1]
    for date, v in _bh_rls_dates.itertuples(index=False)
]

charts = A.vconcat(*charts)
# with A.data_transformers.enable("default"):
#     charts.save(f"../reports/figures/prod_det_uptake_{os}.png")
charts

# %%
buildhub_data_beta.query("disp_vers == '71.0'")

# %% {"jupyter": {"outputs_hidden": true}}
s = dfcb.dvers
s.map(maj_os)

# %% {"jupyter": {"outputs_hidden": true}}
s

# %%
dfcb[:3]

# %%
w = dfcb.query("os == 'Windows_NT' & mvers == 71 & dvers == '71.0'").groupby([c.submission_date, c.dvers]).n.sum().reset_index(drop=0)
w
# [:3]

# %% {"jupyter": {"outputs_hidden": true}}
drops = 'description 	is_security_driven build_number category product'.split()
pd_beta.assign(mvers=lambda x: x.version.map(maj_vers)).query("mvers >= 69").drop(drops, axis=1).sort_values(["date"], ascending=True)

# %%
bh_data.query("chan == 'beta'").reset_index(drop=1)

# %%
maj_json

# %% [markdown]
# # Nightly

# %%
import numpy as np

# %%
dfcn_ = dfcn.assign(bidd=lambda x: x.bid.str[:8])

# %%
dfcn = (
    dfcn_.query("bidd > '201909'")
    .assign(old=lambda x: x.bid.str.len() == 8)
    .assign(bidd=lambda x: np.where(x.old, "20190101", x.bidd))
)

w = dfcn.query("os == 'Windows_NT'").query("submission_date >= '2019-10-01'")

# %%
ww = w.groupby(["submission_date", "bidd"]).n.sum().reset_index(drop=0)

# %%
h = (
    a3.Chart(ww)
    .mark_line()
    .encode(x="submission_date", y="n", color="bidd", tooltip=["bidd", "n", 'submission_date'])
    .properties(height=500, width=700)
    .interactive()
)

# %%
res = (
    dfcn.groupby(['os', c.submission_date])
    .apply(max_bid)
    .pipe(lambda x: DataFrame(x.tolist(), index=x.index))
    .assign(max_dt=lambda x: pd.to_datetime(x.max_bid))
    .reset_index(drop=0)
    .assign(dif=lambda x: x[c.submission_date] - x.max_dt)
    .assign(dif=lambda x: x.dif.astype('timedelta64[D]').astype(int))
)

def vcc(s):
    return s.value_counts(normalize=1).sort_index().cumsum()


# %% [markdown]
# ## Result
# For Linux and Win, 100% of builds reach peak within 2 days. Most peak within the population either the day after, but all the straggler builds peak by the day after *that*.
#
# For Darwin, there's much slower uptake, where many builds might peak _4 days later_.

# %%
dfcn.dvers.drop_duplicates().sort_values(ascending=True)

# %%
res.groupby(['os']).dif.apply(vcc)

# %%
res.dif.value_counts(normalize=1)

# %%
gdf


# %%
def max_bid(gdf):
    ix = gdf.n.idxmax()
    return dict(
        max_bid=gdf.bidd[ix],
        pct=gdf.n.loc[ix] / gdf.n.sum()
    )
    return []
    return np.array([ix, gdf.n.loc[ix] / gdf.n.sum()])
    return gdf.bidd[ix]

max_bid(gdf)

# %%

# %%
ww

# %%
w

# %%
