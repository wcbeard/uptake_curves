# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
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

# %% [markdown]
# ## Product details: get current version

# %%
add_path("/Users/wbeard/repos/dscontrib-moz/src/dscontrib")
import wbeard.buildhub_bid as bh
from requests import get


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


cur_rls_maj = get_recent_release_from_product_details()
min_build_date = get_min_build_date(days_ago=365)

print(min_build_date, cur_rls_maj)


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


maj_json = read_product_details_maj()
prod_details = read_product_details_all()


# %% [markdown]
# ## Buildhub
# - build_ids in past 90 days

# %%
def maj_vers(vers: str) -> int:
    majv, *_ = vers.split(".")
    return int(majv)


def pull_bh_data_rls(min_build_date):
    major_re = re.compile(r"^(\d+)\.\d+$")

    def major(v):
        m = major_re.findall(v)
        if not m:
            return None
        [maj] = m
        return int(maj)

    rls_docs = bh.pull_build_id_docs(min_build_day=min_build_date, channel="release")
    df = bh.version2df(
        rls_docs, major_version=None, keep_rc=False, keep_release=True
    ).assign(chan="release", major=lambda x: x.disp_vers.map(major)).assign(is_major=lambda x: x.major.notnull())
    return df


def pull_bh_data_beta(min_build_date):
    recent_betas_filter = lambda v: maj_vers(v) in [cur_rls_maj, cur_rls_maj + 1]
    beta_docs = bh.pull_build_id_docs(min_build_day=min_build_date, channel="beta")
    return bh.version2df(
        beta_docs, major_version=recent_betas_filter, keep_rc=False, keep_release=True
    ).assign(chan="beta")


buildhub_data_rls = pull_bh_data_rls("2019")

# buildhub_data_rls = pull_bh_data_rls("2019")
buildhub_data_beta = pull_bh_data_beta(min_build_date)

# bh_data = pd.concat([buildhub_data_rls, buildhub_data_beta], ignore_index=True)

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

# %%
pd_rls

# %%
maj_json[::-1]

# %%
buildhub_data_rls.query("is_major").drop_duplicates(subset='disp_vers', keep='last')

# %% [markdown]
# # Compare to actual uptake

# %%
dfc = bq_read('select * from analysis.wbeard_uptake_vers')
len(dfc)

# %% [markdown]
# We have
# * uptake data: `dfc`
# * product details
#     * major: `maj_json` <- https://product-details.mozilla.org/1.0/firefox_history_major_releases.json
#     * all: `pd_rls` <- https://product-details.mozilla.org/1.0/firefox.json
# * buildhub: `buildhub_data_rls`

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
def mk_rls_plot(rls_date, df, version, os="Windows_NT"):
    rls_date = pd.to_datetime(rls_date)
    beg_win = rls_date - pd.Timedelta(days=3)
    end_win = rls_date + pd.Timedelta(days=13)
    w = (
        df.query(f"os == '{os}'")
        .query("submission_date >= @beg_win")
        .query("submission_date <= @end_win")
    )
    w2 = (
        w.groupby(["submission_date", "vers"])
        .n.sum()
        .reset_index(drop=0)
        .assign(dayn=lambda x: x.groupby("submission_date").n.transform("sum"))
        .assign(n_perc=lambda x: x.n / x.dayn * 100)
        .query("n_perc > 1")
    )
    d1 = {'submission_date': rls_date, 'n_perc': 0, 'vers': 'release'}
    d2 = z.merge(d1, dict(n_perc=90))
    w2 = w2.append([d1, d2], ignore_index=True)
    

    h = (
        Chart(w2)
        .mark_line()
        .encode(
            x="submission_date",
            y="n_perc",
            color="vers",
            tooltip=["n", "n_perc", "submission_date", "vers"],
        )
    ).properties(title=str(version))

    hh = (h + h.mark_point()).interactive()
    return w2, hh


# w2, hh = mk_rls_plot("2019-12-03", dfcr)

# %%
maj_json[:3]

# %%

# %%
charts = [mk_rls_plot(date, dfcr, v)[1]
 for date, v in pd_rls[['date', 'version']].query(f"date >= '{dfcr.submission_date.min()}'").itertuples(index=False)]

charts = A.vconcat(*charts)
with A.data_transformers.enable('default'):
    charts.save('../reports/figures/prod_det_uptake.png')
charts

# %%
charts = [mk_rls_plot(date, dfcr, v)[1]
 for date, v in maj_json.query(f"date >= '{dfcr.submission_date.min()}'").itertuples(index=False)]

charts = A.vconcat(*charts)
with A.data_transformers.enable('default'):
    charts.save('../reports/figures/maj_prod_det_uptake.png')
charts

# %%
maj_json
