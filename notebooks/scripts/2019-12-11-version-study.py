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
import altair.vegalite.v3 as a3

add_path("/Users/wbeard/repos/missioncontrol-v2/mc2/")
add_path("/Users/wbeard/repos/missioncontrol-v2/mc2/data")
import data.release_versions as rv

add_path("/Users/wbeard/repos/dscontrib-moz/src/dscontrib")
import wbeard.buildhub_bid as bh
from requests import get
import src.utils.uptake_utes as uu

def proc(df):
    df = df.assign(mvers=lambda x: x.dvers.map(uu.maj_os))
    return df

prod_details = rv.read_product_details()

# %%
# Uptake
dfc = bq_read("select * from analysis.wbeard_uptake_vers")
len(dfc)

# %%
dfcn = bq_read("select * from analysis.wbeard_uptake_vers where chan = 'nightly'")
dfcn = proc(dfcn)
len(dfcn)

# %% [markdown]
# ## Product details: get current version

# %%
(dt.date.today() + pd.Timedelta(days=-2)).strftime('%Y-%m-%d')


# %%
def get_recent_release_from_product_details() -> int:
    """
    Query product-details.mozilla.org/ to get most
    recent major release. (e.g., 71)
    """
    rls_prod_details_json = get(
        "https://product-details.mozilla.org/1.0/firefox_history_major_releases.json"
    ).json()
    rls_prod_details = Series(rls_prod_details_json).sort_values(ascending=True)
    [(cur_rls_vers, _date)] = rls_prod_details[-1:].iteritems()
    cur_rls_maj, *_v = cur_rls_vers.split(".")
    return int(cur_rls_maj)


def get_min_build_date(days_ago=90):
    min_build_datetime = dt.datetime.today() - dt.timedelta(days=days_ago)
    return min_build_datetime.strftime("%Y%m%d")


# cur_rls_maj = get_recent_release_from_product_details()
# min_build_date = get_min_build_date(days_ago=365)

# print(min_build_date, cur_rls_maj)

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
def read_product_details_all():
    pd_url = "https://product-details.mozilla.org/1.0/firefox.json"
    js = pd.read_json(pd_url)
    df = (
        pd.DataFrame(js.releases.tolist())
        .assign(release_label=js.index.tolist())
        .assign(date=lambda x: pd.to_datetime(x.date))
    )
    return df



# Just major release dates

def read_product_details_maj(min_yr=2018):
    """
    Be careful, some versions are non-integer (like '14.0.1').
    The `min_yr` filter should prevent errors.
    """
    maj_json = get(
        "https://product-details.mozilla.org/1.0/firefox_history_major_releases.json"
    ).json()
    df = (
        DataFrame(maj_json.items(), columns=["vers_float", "date"])
        .assign(date=lambda x: pd.to_datetime(x.date))
        .query(f"date > '{min_yr}'")
        .assign(vers_float=lambda x: x.vers_float.astype(float))
        .reset_index(drop=1)
        .assign(vers=lambda x: x.vers_float.astype(int))
    )
    assert (df.vers == df.vers_float).all(), "Some versions aren't integers"
    return df.drop("vers_float", axis=1)


# maj_json = read_product_details_maj()


# %% [markdown]
# ## Beta release

# %%



def get_beta_release_dates(min_build_date="2019", min_pd_date="2019-01-01"):
    """
    Stitch together product details and buildhub
    (for rc builds).
    - read_product_details_all()
    """
    bh_beta_rc = (
        pull_bh_data_beta(min_build_date=min_build_date)
        .rename(columns={"pub_date": "date", "disp_vers": "version"})
        .assign(rc=lambda x: x.version.map(is_rc), src="buildhub")
        .query("rc")[["date", "version", "src"]]
        .sort_values(["date"], ascending=True)
        .drop_duplicates(["version"], keep="first")
    )
    prod_details_all = read_product_details_all()
    prod_details_beta = prod_details_all.query(
        f"category == 'dev' & date > '{min_pd_date}'"
    )[["date", "version"]].assign(src="product-details")

    beta_release_dates = (
        prod_details_beta.append(bh_beta_rc, ignore_index=False)
        .assign(maj_vers=lambda x: x.version.map(lambda x: int(x.split(".")[0])))
        .sort_values(["date"], ascending=True)
        .reset_index(drop=1)
        # Round down datetimes to nearest date
        .assign(date=lambda x: pd.to_datetime(x.date.dt.date))
        .astype(str)
    )
    return beta_release_dates


# def is_rc(v):
#     """
#     Is pattern like 69.0, rather than 69.0b3
#     """
#     return "b" not in v


_bh_rls_dates = buildhub_data_beta.rename(
    columns={"pub_date": "date", "disp_vers": "version"}
)[["date", "version"]].query(f"date >= '{dfcb.submission_date.min()}'")

# bh_beta_rc = get_beta_release_dates()
beta_release_dates = get_beta_release_dates()

# %%
beta_release_dates = rv.get_beta_release_dates(min_build_date='2019', min_pd_date='2019-01-01')
rv.latest_n_release_beta(beta_release_dates, '2019-12-02')


# %%
def latest_n_release_beta(beta_release_dates, sub_date, n_releases: int=1):
    """
    Given dataframe with beta release dates and a given
    submission date (can be from the past till today), return the
    `n_releases` beta versions that were released most recently.
    beta_release_dates: df[['date', 'version', 'src', 'maj_vers']]
    """
    beta_release_dates = beta_release_dates[['date', 'version', 'src', 'maj_vers']]
    latest = (
        beta_release_dates
        # Don't want versions released in the future
        .query("date < @sub_date")
        .sort_values(["date"], ascending=True)
        .iloc[-n_releases:]
    )
    
    return latest

latest_n_release_beta(beta_release_dates, '2019-12-04', n_releases=3)

# %%
pd_beta = prod_details.query("category == 'dev' & date > '2019'")

# %% {"jupyter": {"outputs_hidden": true}}
pd_beta = pull_bh_data_beta()

# %% [markdown]
# ## Buildhub
# - build_ids in past 90 days

# %% [markdown]
# ## Beta

# %%
# !cp /Users/wbeard/repos/dscontrib-moz/src/dscontrib/wbeard/buildhub_bid.py '../src/data/'

# %%
import src.data.buildhub_utils as bh

# %%
bh


# %%
def pull_bh_data_beta(min_build_date):
    # recent_betas_filter = lambda v: maj_vers(v) in [cur_rls_maj, cur_rls_maj + 1]
    beta_docs = bh.pull_build_id_docs(min_build_day=min_build_date, channel="beta")
    return bh.version2df(
        beta_docs,
        # major_version=recent_betas_filter,
        keep_rc=False,
        keep_release=True,
    ).assign(chan="beta")


# %% [markdown]
# ## Release

# %%
import src.data.release_dates as rd

def maj_vers(vers: str) -> int:
    majv, *_ = vers.split(".")
    return int(majv)


buildhub_data_rls = rd.pull_bh_data_rls("2019")
buildhub_data_beta = rd.pull_bh_data_beta('2019')

# bh_data = pd.concat([buildhub_data_rls, buildhub_data_beta], ignore_index=True)

# %%
buildhub_data_beta


# %%
def check_1to1_mapping(s1, s2):
    dmap = dict(zip(s1, s2))
    s2_should_be = s1.map(dmap)
    ne = s2_should_be != s2
    
    assert ne.sum() == 0, 'Not 1-1 mapping!'
    
check_1to1_mapping(bh_data.build_id, bh_data.disp_vers)
check_1to1_mapping(bh_data.build_id, bh_data.pub_date)

# %%
pd_rls = prod_details.query("category == 'major' & date > '2019'")
pd_beta = prod_details.query("category == 'dev' & date > '2019'")

# %%
pd_rls

# %%
maj_json[::-1]

# %%
buildhub_data_rls.query("is_major").drop_duplicates(subset='disp_vers', keep='last')

# %% [markdown]
# # Compare to actual uptake
# ## Release

# %%

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
dfcb[:3]

# %%

# %%
maj_json[:3]

# %%

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
def mk_rls_plot(rls_date, df, version, vers_col="vers", os="Windows_NT", min_max_pct=10):
    rls_date = pd.to_datetime(rls_date)
    beg_win = rls_date - pd.Timedelta(days=3)
    end_win = rls_date + pd.Timedelta(days=13)
    w = (
        df.query(f"os == '{os}'")
        .query("submission_date >= @beg_win")
        .query("submission_date <= @end_win")
    )
    w2 = (
        w.groupby(["submission_date", vers_col])
        .n.sum()
        .reset_index(drop=0)
        .assign(dayn=lambda x: x.groupby("submission_date").n.transform("sum"))
        .assign(n_perc=lambda x: x.n / x.dayn * 100)
        .assign(n_perc_vers=lambda x: x.groupby(vers_col).n_perc.transform("max"))
        .query("n_perc_vers > @min_max_pct")
    )
    d1 = {"submission_date": rls_date, "n_perc": 0, vers_col: "release"}
    d2 = z.merge(d1, dict(n_perc=90))
    w2 = w2.append([d1, d2], ignore_index=True)

    h = (
        Chart(w2)
        .mark_line()
        .encode(
            x="submission_date",
            y="n_perc",
            color=vers_col,
            tooltip=["n", "n_perc", "submission_date", vers_col],
        )
    ).properties(title=str(version))

    hh = (h + h.mark_point()).interactive()
    return w2, hh


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
