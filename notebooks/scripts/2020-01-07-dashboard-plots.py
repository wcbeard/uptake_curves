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

# %%

# %% [markdown]
# ## Beta release

# %%
# import data.queries as qs

@mem.cache
def bq_read_cache(*a, **k):
    return bq_read(*a, **k)




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
import src.uptake_plots as up

# %%
# prod_details = rv.read_product_details_all()
pd_rls = prod_details.query("date > '2019' & category in ('major', 'stability')").pipe(
    lambda x: x[~x["release_label"].str.endswith("esr")]
)
# pd_beta = prod_details.query("category == 'dev' & date > '2019'")

# %%
dfcr = dfcr.copy()
dfcr = up.combine_uptake_dates_release(dfcr, pd_rls, vers_col="dvers")

# %%
# dfcr = dfc.query("chan == 'release'")
w = dfcr.query("os == 'Windows_NT'")

# %%
w[:3]

# %%
pdf = up.format_os_df_plot(w, pub_date_col=c.rls_date).query("submission_date > '2019-10-01'")

# %%
import altair.vegalite.v3 as A
Chart = A.Chart

# %%
p_all = up.os_plot_base_release(pdf, color="vers:O", A=A)
p_all.interactive()
