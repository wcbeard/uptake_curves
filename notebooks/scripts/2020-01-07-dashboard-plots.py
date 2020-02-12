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
add_path('/Users/wbeard/repos/dscontrib-moz/src/dscontrib/')
import wbeard.buildhub_utils as dbh
# # add_path("/Users/wbeard/repos/missioncontrol-v2/mc2/")
# # add_path("/Users/wbeard/repos/missioncontrol-v2/mc2/data")
# # import data.release_versions as rv

# add_path("/Users/wbeard/repos/dscontrib-moz/src/dscontrib")
# # import wbeard.buildhub_bid as bh
# from requests import get

# %%
import uptake.data.buildhub_utils as bh
import uptake.utils.uptake_utes as uu
import uptake.data.release_dates as rv
import altair.vegalite.v3 as a3

# def proc(df):
#     df = df.assign(mvers=lambda x: x.dvers.map(uu.maj_os))
#     return df

# %%
# Uptake
dfc = bq_read("select * from analysis.wbeard_uptake_vers")
print(len(dfc))

# %%

# %% [markdown]
# ## Beta release

# %% [raw]
# # import data.queries as qs
#
# @mem.cache
# def bq_read_cache(*a, **k):
#     return bq_read(*a, **k)

# %% [raw]
# _bh_rls_dates = buildhub_data_beta.rename(
#     columns={"pub_date": "date", "disp_vers": "version"}
# )[["date", "version"]].query(f"date >= '{dfcb.submission_date.min()}'")
#
# # bh_beta_rc = get_beta_release_dates()
# beta_release_dates = get_beta_release_dates()

# %% [raw]
# beta_release_dates = rv.get_beta_release_dates(min_build_date='2019', min_pd_date='2019-01-01')
# rv.latest_n_release_beta(beta_release_dates, '2019-12-02', n_releases=3)

# %% [raw]
# # pd_beta = prod_details.query("category == 'dev' & date > '2019'")
# # pd_beta = pull_bh_data_beta()

# %% [markdown]
# # Uptake plots

# %% [markdown]
# ## Release

# %% [markdown]
# We have
# * uptake data: `dfc`
# * product details
#     * major: `maj_json` <- https://product-details.mozilla.org/1.0/firefox_history_major_releases.json
#     * all: `pd_rls` <- https://product-details.mozilla.org/1.0/firefox.json
# * buildhub: `buildhub_data_rls`

# %%
import altair.vegalite.v3 as A
Chart = A.Chart
from uptake.plot import uptake_plots as up
from uptake.plot import plot_upload as plu

# %%
prod_details = rv.read_product_details_all()
dfc = bq_read("select * from analysis.wbeard_uptake_vers")

# %%
# dfcr, dfcb, dfcn = process_raw_channel_counts(dfc, prod_details)
dfcr, dfcb, dfcn = plu.main()

# %%

# %%
dfcr[:3]

# %%
pd.options.display.width = 220
pd.options.display.max_columns = 20

# %%
df_plottable.submission_date.min()

# %%
res = format_channel_data(dfcr, channel="release")
len(res)

# %%
import bq_utils as bq

def format_all_channels_data(dfr, dfb, dfn, channels_months_ago=(7, 3, 3), sub_date: str = None):
    def get_min_date(months_ago):
        date = pd.to_datetime(sub_date) if sub_date else dt.datetime.today()
        return (date - pd.Timedelta(days=3 * months_ago)).strftime(bq.SUB_DATE)

    r_min_date, b_min_date, n_min_date = map(get_min_date, channels_months_ago)
    df_plottable = pd.concat(
        [
            plu.format_channel_data(dfr, min_date=r_min_date, channel="release"),
            plu.format_channel_data(dfb, min_date=b_min_date, channel="beta"),
            plu.format_channel_data(dfn, min_date=n_min_date, channel="nightly"),
        ],
        sort=False,
    )
    return df_plottable

del format_all_channels_data
# df_plottable = format_all_channels_data(dfcr, dfcb, dfcn)

# %%
def look_at_special_mostly_null_fields():
    def _app(gdf):
        return gdf.notnull().mean()

    def drop_boring(nulls):
        nm = nulls.mean(axis=1)
        boring = nulls.sub(nm, axis=0).eq(0).all(axis=1)
        return nulls[~boring].round(1)

    nulls = df_plottable.groupby(["channel"]).apply(_app).T

    return drop_boring(nulls)


# look_at_special_mostly_null_fields()

# %% [markdown]
# # Plots from BQ
# - [ ] discrepancy in dates
#     - my new fun has much smaller date range than original

# %%
del df_plottable

# %%
df_plottable_yest = plu.main(sub_date='2020-02-05')
df_plottable_yest.submission_date.max()

# %%
df_plottable_tod = plu.main(sub_date='2020-02-09')
df_plottable_tod.submission_date.max()

# %%
df_plottable_to_upl = plu.main(sub_date='2020-02-09')

# %%
df_plottable_to_upl.submission_date.value_counts(normalize=0)

# %%
df_plottable_tod.submission_date.drop_duplicates()

# %%
print(q)

# %%
q = f"select distinct submission_date as date from analysis.wbeard_uptake_plot_test where submission_date > '{plu.to_sql_date(df_plottable_tod.submission_date.min())}'"
distinct_existing_dates = bq_read(q).date.dt.tz_localize(None)

# %%
missing = df_plottable_tod.pipe(lambda x: x[~x.submission_date.isin(distinct_existing_dates)])

# %%
df_plottable_to_upl = plu.main(sub_date='2020-02-09')

# %%
len(df_plottable_to_upl)

# %%
df_plottable_to_upl

# %%
import uptake.plot.plot_upload as plu
import uptake.plot.uptake_plots as up

# dfpr = df_plottable.query("channel == 'release'")
# dfpr = dfpr.rename(columns={'nth_recent_release': 'nrro'}).reset_index(drop=1)
# .drop(['nth_recent_release'], axis=1)

# %%
df_plottable_yest[:3]

# %%
creds = bq.get_creds()

# %%
df_plottable.submission_date.max()

# %% [markdown]
# ## Render from BQ

# %%

# %%
df_plottable_yest.to_gbq('analysis.wbeard_uptake_plot_test', project_id='moz-fx-data-derived-datasets', credentials=creds, if_exists='replace')

# %%
d1 = df_plottable_tod[['vers', 'submission_date', 'os', 'channel', 'n']]
d2 = df_plottable_yest[['vers', 'submission_date', 'os', 'channel', 'n']].rename(columns={'n': 'n2'})

mgd = d1.merge(d2, on=['vers', 'submission_date', 'os', 'channel'], how='left').assign(new=lambda x: x.n2.isnull())
mgd[:3]

# %%
df_pl

# %% {"jupyter": {"outputs_hidden": true}}
mgd.query("new")
# .query("n2 == n2")

# %%
d1.submission_date.max()

# %%
d2.submission_date.max()

# %%
len(df_plottable_yest)

# %%

# %%
from uptake.plot import embed_html as eht

# %%
# import uptake.plot.embed_html


bq_read_cache = bq.mk_bq_reader(cache=True)
dl_pl_rls = eht.download_channel_plot('release', dt.date.today(), bq_read=bq_read_cache)
od2 = up.generate_channel_plot(dl_pl_rls, A, min_date="2019-06-01", channel='release', separate=True)
eht.render_channel(win=od['Windows_NT'], mac=od['Darwin'], linux=od['Linux'], channel=channel)

# %%
# eht.main??

# %%
eht.main(sub_date='2020-02-10')

# %%
dl_pl_rls[:3]

# %%

# %%
# import uptake.plot.embed_html as emb

od = up.generate_channel_plot(dfpr, A, min_date="2019-06-01", channel='release', separate=True)
emb.render_channel(win=od['Windows_NT'], mac=od['Darwin'], linux=od['Linux'], channel=channel)

# %%
dfpr[:3]

# %%
dl_pl_rls[:3]

# %%
len(dl_pl_rls)

# %%
# from uptake.plot.plot_download import templ_sql_rank

dest_table="analysis.wbeard_uptake_plot_test"
sub_date = dt.date.today()
min_sub_date = sub_date - pd.Timedelta(days=30*5)


sql_rank = templ_sql_rank.format(
    channel='release',
min_sub_date=plu.to_sql_date(min_sub_date),
max_sub_date=plu.to_sql_date(sub_date),
)

print(sql_rank)

# %%
sql_rank = '''with base as (
select *
from `analysis.wbeard_uptake_plot_test` u
)

, unq_vers as (
select
  os
  , vers
  , channel 
  , rank() over (partition by os, channel order by min(b.build_ids) desc) as rank
from base b
where not b.old_build
group by 1, 2, 3
)

, builds as (
select
  b.*
  , rank
from base b
left join unq_vers using (os, vers, channel)
)


select *
from builds
--where rank is not null
order by os, channel, rank
'''

df_pl_dl_ = bq_read(sql_rank)

# %%
df_pl_dl = df_pl_dl_.assign(
    submission_date=lambda x: x.submission_date.dt.tz_localize(None),
    vers_min_date_above_npct=lambda x: x.vers_min_date_above_npct.dt.tz_localize(None),
).drop(["nth_recent_release", 'latest_vers'], axis=1).rename(columns={'rank': 'nth_recent_release'})


# %%
len(df_pl_dl.query("~old_build"))

# %%
df_pl_dl.groupby(c.submission_date).size()

# %%
df_pl_dl[:3]

# %%

# %%
df_pl_dl[:3]

# %%
up.generate_channel_plot(
    df_pl_dl.query("channel == 'release'"), A, min_date="2019-10-01", channel="release", separate=False
)

# %%
df_pl_dl[:3]

# %%
df_pl_dl.query("latest_vers")

# %%
dfo = df_plottable.sort_values(
    ["os", "channel", "vers", "submission_date"], ascending=True
).reset_index(drop=1)
df2 = (
    df_pl_dl.sort_values(["os", "channel", "vers", "submission_date"], ascending=True)
    .drop(["rank"], axis=1)
    .assign(
        is_major=lambda x: x.is_major.fillna(float('nan'))
    )
    .reset_index(drop=1)
)

# %%
dfo.query("RC != RC").channel.value_counts(normalize=0)


# %% {"jupyter": {"outputs_hidden": true}}
def compare_dtypes(d1, d2):
    dtps = (
        DataFrame({"d1": d1.dtypes, "d2": d2.dtypes})
        .reset_index(drop=0)
        .rename(columns={"index": "colname"})
        .assign(same=lambda x: x.d1 == x.d2,)
    )
    return dtps

def compare_srs(s1, s2):
    dd = DataFrame({'s1': s1, 's2': s2})
    dd.query("s1 != s2")
#     .query("o == o").applymap(type)
    return dd


# compare_dtypes(dfo, df2)
compare_srs(dfo.is_major, df2.is_major)

# %%
assert_frame_equal(dfo, df2)

# %%
gb = (
    df_pl_dl.query("nth_recent_release == nth_recent_release")[
        ["nth_recent_release", "rank", "n"]
    ]
    .assign(eq=lambda x: x.nth_recent_release == x["rank"])
    .groupby(["eq", "nth_recent_release", "rank"])
    .n.sum()
    .reset_index(drop=0)
)


# %%
gb.groupby(['eq']).n.sum().pipe(lambda x: x.div(x.sum()))

# %%
df_plottable[:3]

# %%
df_pl_dl.submission_date.dt.tz_localize(None)

# %%
dfpr[:3]

# %%
# dfpr.query("nth_recent_release == nth_recent_release")[:3]
dfpr[:3]

# %%
dfpr.query("os == 'Darwin' & vers == '71.0'").drop_duplicates([c.build_ids, c.min_build_id])

# %%
dfpr2 = merge_channel_ordered_versions(dfpr)

# %%
len(recent_version_order)

# %%
dfpr2[:3]

# %%
verss = dfpr2.dropna(axis=0, subset=['nrro']).query("~old_build")[['nrro', c.recent_version_order]]

# %%
diffs = verss.pipe(lambda x: x[x.nrro != x.recent_version_order])
diffs[:3]

# %%
pd.to_datetime('2019-11-01') + pd.Timedelta(days=28)

# %%
lin = dfpr2.query("os == 'Linux'")

# %%
add_ones = lambda df: df.reset_index(drop=0).reset_index(drop=0).assign(index=lambda x: x['index'] + 1)
lin.groupby(['nrro', 'vers']).size().pipe(add_ones)

# %%
lin.groupby(['recent_version_order', 'vers']).size().pipe(add_ones)

# %%
dfpr.query("os == 'Linux'").query("nrro in (16, 17, 18)").drop_duplicates('vers')

# %%
dfpr2.query("os == 'Linux'").query("nrro in (16, 17, 18, 19, 20)").drop_duplicates('vers').sort_values(["nrro"], ascending=True)

# %%
dfpr2.loc[diffs.index].query("recent_version_order == 18.0")

# %%
dfpr.groupby(['vers', 'os'])[c.vers_min_date_above_npct].min().unstack().fillna(0)


# %%
# chs_rls = up.generate_channel_plot(dfcr, A, min_date="2019-06-01", channel='release')
# od = up.generate_channel_plot_full(dfcr, A, min_date="2019-06-01", channel='release', separate=True)

# %%

# %%
def f(od, channel = 'release'):
    for os, h in od.items():
        out = json_base / f'{channel}-{os}.json'
        with open(out,'w') as fp:
            h.save(fp, format='json')
    f.h = h
    
f(od)
h = f.h

# %%
channel = 'release'
os = 'Windows_NT'
json_base = Path('/Users/wbeard/repos/uptake_curves/reports/channel_html')
out = json_base / f'{channel}-{os}.json'

# %%

# %%

# %%
od['Windows_NT']

# %%
df_plottable.submission_date.min()

# %%
dfcr.submission_date.min()

# %%
chs_rls

# %%
key = ['vers', 'submission_date', 'os']
res.nunique()[key]

# %%
res.groupby(['vers', c.submission_date, 'os']).size().value_counts(normalize=0)


# %% [markdown]
# # Plots dev

# %%
def generate_channel_plot(df, A, min_date="2019-10-01", channel="release"):
    channel_release_disp_days = dict(
        release=30,
        beta=10,
        nightly=7,
    )
    max_days_post_pub = channel_release_disp_days[channel]
    
    od = OrderedDict()
    for os in ("Windows_NT", "Darwin", "Linux"):
        osdf = df.query("os == @os")
        pdf = (
            up.format_os_df_plot(osdf, pub_date_col=c.rls_date, channel=channel)
            .query(f"submission_date > '{min_date}'")
            .assign(os=os)
            .query("nth_recent_release == nth_recent_release")
        )
        ch = (
            up.os_plot_base_release(
                pdf,
                color="vers:O",
                separate=False,
                A=A,
                channel=channel,
                max_days_post_pub=max_days_post_pub,
            )
            .properties(height=500, width=700, title=f"OS = {os}")
            .interactive()
        )
        od[os] = ch
    generate_channel_plot.od = od
    return (
        A.vconcat(*od.values())
        .resolve_scale(color="independent")
        .resolve_axis(y="shared")
    )


chs_rls = generate_channel_plot(dfcr, A, min_date="2019-06-01")
chs_rls

# %%
import pathlib

absl = pathlib.Path('.').absolute()

# %%
chs_beta = generate_channel_plot(dfcb, A, min_date="2019-10-01", channel='beta')
chs_beta

# %% [markdown]
# ## Beta

# %%
# dfcb.assign(rls_date=lambda x: x.rls_date.astype(str)).groupby(
#     ["dvers", "rls_date"]
# ).n.sum().reset_index(drop=0).query("rls_date == 'nan'")

beta_dates = rv.get_beta_release_dates(min_build_date="2019", min_pd_date="2019-01-01")

# %%
dfcb = dfc.query("chan == 'beta'").copy()
dfcb = up.combine_uptake_dates_release(dfcb, beta_dates, vers_col="dvers").drop('vers', axis=1).rename(columns={'dvers': 'vers'})

# %%
dfcb[:3]

# %%
dfcb[:3]

# %%
dfcr


# %%
# w = dfcb.query("os == 'Windows_NT'")
# pdf = up.format_os_df_plot(w, pub_date_col=c.rls_date, channel='beta').query("submission_date > '2019-10-01'")

# min_days_post_sub = pdf.groupby('vers').days_post_pub.transform('min')
# pdf.loc[min_days_post_sub == -1, 'days_post_pub'] += 1

# %%
def generate_beta_plot(df, min_date="2019-10-01"):
    od = OrderedDict()
    for os, osdf in df.groupby("os"):
        pdf = (
            up.format_os_df_plot(osdf, pub_date_col=c.rls_date, channel="beta")
            .query(f"submission_date > '{min_date}'")
            .assign(os=os)
            .query("nth_recent_release == nth_recent_release")
        )
        ch = (
            up.os_plot_base_release(
                pdf, color="vers:O", separate=False, A=A, channel="beta"
            )
            .properties(height=500, width=700, title=f"OS = {os}")
            .interactive()
        )
        od[os] = ch
    generate_beta_plot.od = od
    return A.concat(*od.values()).resolve_scale(color="independent")


chs_beta = generate_beta_plot(dfcb)
# chs_beta

# %% [markdown]
# ## Nightly
# - [X] determine version units
#     - bid[:8]
# - [ ] add `rls_date`

# %%

# %%
dfcn = (
    dfc.query("chan == 'nightly'")
    .copy()
    .assign(build_day=lambda x: x.bid.str[:8])
    .assign(rls_date=lambda x: pd.to_datetime(x.build_day))
    .drop('vers', axis=1)
    .rename(columns={'build_day': 'vers'})
)

# %%
# channel='nightly'
# _os = 'Windows_NT'
# w = dfcn.query(f"os == '{_os}'")
# pdf = up.format_os_df_plot(w, pub_date_col=c.rls_date, channel=channel).query("submission_date > '2019-10-01'").assign(os=_os)
# del w
# del pdf, _os, channel

# %%
from altair_saver import save

with A.data_transformers.enable('default'):
    save(ch_nightly, '../reports/figures/os_nightly.html', inline=True)

# %%
ch_nightly = generate_channel_plot(dfcn, A, min_date="2019-10-01", channel="nightly")

# %%
ch_nightly

# %%
add_path('/Users/wbeard/repos/dscontrib-moz/src/dscontrib/')
import wbeard.buildhub_utils as dbh
