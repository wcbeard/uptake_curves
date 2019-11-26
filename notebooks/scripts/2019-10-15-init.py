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
# # Buildhub info

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
min_build_date = get_min_build_date(days_ago=150)

print(min_build_date, cur_rls_maj)


# %% [markdown]
# ## Buildhub
# - build_ids in past 90 days

# %%
def maj_vers(vers: str) -> int:
    majv, *_ = vers.split(".")
    return int(majv)


def pull_bh_data_rls(min_build_date):
    recent_rls_filter = lambda v: maj_vers(v) in [
        cur_rls_maj - 2,
        cur_rls_maj - 1,
        cur_rls_maj,
    ]
    rls_docs = bh.pull_build_id_docs(min_build_day=min_build_date, channel="release")
    return bh.version2df(
        rls_docs, major_version=recent_rls_filter, keep_rc=False, keep_release=True
    ).assign(chan="release")


def pull_bh_data_beta(min_build_date):
    recent_betas_filter = lambda v: maj_vers(v) in [cur_rls_maj, cur_rls_maj + 1]
    beta_docs = bh.pull_build_id_docs(min_build_day=min_build_date, channel="beta")
    return bh.version2df(
        beta_docs, major_version=recent_betas_filter, keep_rc=False, keep_release=True
    ).assign(chan="beta")



buildhub_data_rls = pull_bh_data_rls(min_build_date)
buildhub_data_beta = pull_bh_data_beta(min_build_date)

bh_data = pd.concat([buildhub_data_rls, buildhub_data_beta], ignore_index=True)
assert (
    bh_data.groupby([c.build_id, "pub_date"])
    .size()
    .reset_index(drop=0)
    .groupby(c.build_id)
    .size()
    .eq(1)
    .all()
), "There's not a 1-1 mapping between build_id and pub_date!"


# %%
def check_1to1_mapping(s1, s2):
    dmap = dict(zip(s1, s2))
    s2_should_be = s1.map(dmap)
    ne = s2_should_be != s2
    
    assert ne.sum() == 0, 'Not 1-1 mapping!'
    
check_1to1_mapping(bh_data.build_id, bh_data.disp_vers)

# %%
buildhub_data_rls[:]

# %% [markdown]
# # Load clients_daily

# %%
import data.queries as qs


@mem.cache
def bq_read_cache(*a, **k):
    return bq_read(*a, **k)


def sub_days_null(s1, s2, fillna=None):
    diff = s1 - s2
    not_null = diff.notnull()
    nn_days = diff[not_null].astype("timedelta64[D]").astype(int)
    diff.loc[not_null] = nn_days
    return diff


dfa_ = bq_read_cache(qs.allq.format(min_sub_date="2019-07-01", min_build_id=20190601))
print(len(dfa_))


# %%
def join_bh_cdaily(cdaily_df, bh_data):
    cdaily_df2 = cdaily_df.merge(
        bh_data.rename(
            columns={
                "disp_vers": "bh_dvers",
                "pub_date": "bh_pub_date",
                "build_id": "bid",
            }
        ),
        on=["chan", "bid"],
        how="left",
    )
    assert len(cdaily_df2) == len(cdaily_df), "Duplicate rows in bh_data?"
    return cdaily_df2

dfa = join_bh_cdaily(dfa_, bh_data).assign(same_dv=lambda x: x.dvers == x.bh_dvers).assign(
    day_chan_os_sum = lambda x: x.groupby(['submission_date', 'chan', 'os']).n.transform('sum')
)
dfa['days_post_pub'] = sub_days_null(dfa.submission_date, dfa.bh_pub_date)

assert dfa.days_post_pub.min() >= -1, "We shouldn't get submissions before it's published"
print(len(dfa))


# %% [markdown]
# ## Prototype 'release' plots
# - choices: restrict build_id to n days later?
#     - but we want it in the denominator
#     
# - days till next release
# - sometimes buildhub publish date is several days before it actually goes out
#     - we'll have to use heuristics :(
#     - additionally, filter out 'previous' versions so as to not plot their rise
#         - ideas: buildid needs to be not too long ago?
#             - this will require knowledge of the plotting window

# %%
# _cand = dfr.query("bid > '20190701' & os == 'Windows_NT'")
# _cand.bh_pub_date.isnull().mean()
# _cand.groupby(_cand.bh_pub_date.isnull()).n.sum()
# _cand

# %%
def is_major(vers: Series):
    return vers.str.count(r"\.").eq(1)


dfr = (
    dfa.query("chan == 'release'")
    .sort_values(
        ["bh_pub_date", "submission_date", "chan", "os", "bid", "vers"], ascending=True
    )
    .reset_index(drop=1)
    .assign(major=lambda x: is_major(x.vers))
)


# %% [markdown]
# #### First, Windows

# %%
def order_bids(bids):
    return "\n".join(bids)


def arr_itg(n):
    def f(srs):
        return srs.values[n]

    return f


def get_vers_first_day_above_n_perc_mapping(df, min_pct=0.01):
    df = df[["n_pct", "submission_date", "vers"]]
    return df.query("n_pct > @min_pct").groupby("vers").submission_date.min().to_dict()


# %%
def format_os_df_plot(os_df):
    """
    - contains data for single os & channel
    - collapse all build_id's into single version
    """
    _cs_sort = ["vers", "submission_date", "n"]
    _cs_sort_all = _cs_sort + ["bid", "bh_pub_date", "day_chan_os_sum"]
    latest_version = (
        os_df[["bid", "vers"]]
        .drop_duplicates()
        .sort_values(by=["bid"], ascending=False)
        .vers.iloc[0]
    )

    gb = (
        os_df[_cs_sort_all]
        .sort_values(_cs_sort, ascending=[True, True, False])
        .assign(
            bids=lambda x: x.bid,
            vers_pub_date=lambda x: x.groupby(["vers"]).bh_pub_date.transform("min"),
            day_chan_os_sum_min=lambda x: x.day_chan_os_sum,
        )
        .groupby(["vers", "submission_date"], sort=False)
    )

    pdf = (
        gb.agg(
            {
                "bids": order_bids,
                "n": "sum",
                "day_chan_os_sum": "max",
                "day_chan_os_sum_min": "min",
            }
        )
        .reset_index(drop=0)
        .assign(
            n_pct=lambda x: x.n / x.day_chan_os_sum,
            latest_vers=lambda x: x.vers.eq(latest_version),
            is_major=lambda x: is_major(x.vers),
        )
        .assign(
            vers_min_sday_npct=lambda x: x.vers.map(
                get_vers_first_day_above_n_perc_mapping(x, 0.01)
            )
        )
        .assign(
            days_post_pub=lambda x: sub_days_null(
                x.submission_date, x.vers_min_sday_npct
            )
        )
    )

    assert pdf.eval(
        "day_chan_os_sum_min == day_chan_os_sum"
    ).all(), "hopefully single values at level of grouping"
    return pdf


min_plottable_bid = bh_data.query("chan == 'release'").build_id.min()
pdf = format_os_df_plot(os_df.query("bid >= @min_plottable_bid"))


# %%
def os_plot_base(df, color="vers:O", separate=False):
    version_opacity = A.Opacity(
        "latest_vers:O", scale=A.Scale(domain=[True, False], range=[1, 0.6])
    )
    major_shape = A.Shape(
        "is_major", scale=A.Scale(domain=[True, False], range=['square', "triangle-up"])
    )
    h = (
        Chart(df.query("days_post_pub <= 14 & days_post_pub > -2"))
        .mark_line(strokeDash=[5, 2])  # , strokeOpacity=.6
        .encode(
            x=A.X("days_post_pub", axis=A.Axis(title="Days after release")),
            y=A.Y("n_pct", axis=A.Axis(format="%", title="Percent uptake")),
            strokeOpacity=version_opacity,  #'latest_vers:O',
            color="vers:O",
            shape=major_shape,
            tooltip=[
                "vers",
                "days_post_pub",
                "submission_date",
                "vers_min_sday_npct",
                "bids",
            ],
        )
    )
    if separate:
        return h, h.mark_point()
    return h + h.mark_point()


p_all = os_plot_base(pdf, color="vers:O")
p_all.interactive()

# %%
rls_plots = {}
min_plottable_bid = bh_data.query("chan == 'release'").build_id.min()


def os_plot(os_name, os_df):
    pdf = format_os_df_plot(os_df.query("bid >= @min_plottable_bid"))
    p_all = os_plot_base(pdf, color="vers:O").properties(title=f"OS={os_name}")
    return p_all

for os, os_df in dfr.groupby("os"):
    rls_plots[os] = os_plot(os, os_df)

rls_plots["Windows_NT"]

# %%
release_row = rls_plots["Windows_NT"] | rls_plots["Darwin"] | rls_plots["Linux"]
release_row

# %%
[
    for (bid, pubdate, os), gdf in os_df.groupby(['bid', c.bh_pub_date, 'os'])
    for os, os_df in dfr.groupby('os')
]

# %%
for os, os_df in dfr.groupby('os'):
    for (bid, pubdate, os), gdf in os_df.groupby(['bid', c.bh_pub_date, 'os']):
        gdf.set_index('days_post_pub').eval('n / day_chan_os_sum').plot(style='o-')
        print(os)
    #     break
    #     if gdf.n.sum() < 1e3:
    #         continue
    if os == 'Windows_NT':
        break

# %%
for (bid, pubdate, os), gdf in dfr.groupby(['bid', c.bh_pub_date, 'os']):
    print(os)
#     break
#     if gdf.n.sum() < 1e3:
#         continue
    if os == 'Windows_NT':
        break

# %%
gdf

# %%
gdf.set_index('days_post_pub').eval('n / day_chan_os_sum').plot(style='o-')

# %%

# %%
gdf

# %%

# %%
gdf

# %%
dfr.groupby(['bid', c.bh_pub_date]).n.sum()

# %%
dfa.groupby(['bh_pub_date', 'chan', 'os', 'vers', 'bid', 'submission_date']).n.sum()

# %%
# min_dates = dfa.groupby(['same_dv', 'chan', 'os', 'vers'])[[c.submission_date, c.bh_pub_date]].min().reset_index(drop=0)
# min_dates.to_csv('/tmp/x.csv')
# # !open '/tmp/x.csv'

# %%

# %%
dfa.groupby(['chan', 'os', 'same_dv']).n.sum().unstack().fillna(0).astype(int).assign(
     Perctrue=lambda x: x[True].div(x[True] + x[False]).mul(100).round(2),
 )

# %%
dfa.groupby('bh_pub_date').n.sum()

# %%
dfa.query("chan == 'release'")[:3]

# %%
# min(bid) is clipped in SQL
top_msg_bids = dfa.query("chan == 'release'").pipe(lambda x: x[x.bid > x.bid.min()]).query(
    "~same_dv"
).groupby("bid").n.sum().pipe(lambda x: x[x > 10_000]).index.tolist()

# %%
versions = dict(line.split('==') for line in pf.splitlines() if '==' in line)
versions = z.keymap(str.lower, versions)
versions[name]

# %%
# .query("bid in @top_msg_bids")
dfa.query("chan == 'release'").groupby("os vers dvers bid same_dv".split()).n.sum().unstack().fillna(0).astype(int).assign(tot=lambda x: x[True] + x[False])    .assign(
     Perctrue=lambda x: x[True].div(x[True] + x[False]).mul(100).round(2),
 ).sort_values(["tot"], ascending=False)[:30]

# %%
len(dfa2)

# %%
len(dfa)

# %%
dfa = dfa.assign(rls_bid=lambda x: x.bid.isin(buildhub_data_rls.build_id))


dfa.groupby(['chan', 'os', 'rls_bid']).n.sum().unstack().fillna(0).astype(int).assign(
     Perctrue=lambda x: x[True].div(x[True] + x[False]).mul(100).round(2),
     Perc=lambda x: (x[True] + x[False]).pipe(lambda s: s / s.sum()).mul(100).round(2)
 )

# %%

# %%



# %%
buildhub_data_beta

# %%
'Windows_NT Darwin Linux'.split()

# %% [markdown]
# ## Release

# %%
dfr = dfa.query("chan == 'release' and os == 'Windows_NT'")

# %%
dfr.os.value_counts(normalize=0)

# %%
dfr.dvers.value_counts(normalize=0)

# %%
dfrbh.pull_build_id_docs(min_build_day=min_build_date, channel="beta")

# %%
wn_ = bq_read_cache(winq).assign(submission_date=lambda x: pd.to_datetime(x.submission_date))

# %%
wn_[:3]

# %%
versions = ["68.0", "69.0", "69.0.1", "69.0.2"]
versions = [
    ("68.0", "2019-07-09"),
    ("69.0", "2019-09-03"),
    ("69.0.1", "2019-09-18"),
    ("69.0.2", "2019-10-03"),
]

# %%
cur_vers

# %% {"jupyter": {"outputs_hidden": true}}
wn.query('cur_vers').groupby(['submission_date', 'cur_vers']).n.sum()

# %%
cur_vers

# %%
agg

# %%
# dct = {}
aggs = []

for cur_vers, st_date in versions:
    wn = wn_.assign(cur_vers=lambda x: x[c.app_version] == cur_vers).query('submission_date >= @st_date')
#     print('\n\n', cur_vers)
#     print(wn.query('cur_vers').groupby(['submission_date', 'cur_vers']).n.sum())
    agg_ = wn.groupby(['submission_date', 'cur_vers']).n.sum().unstack().fillna(0).rename(columns={True: "cur", False: "other"}).assign(perc=lambda x: x.cur.div(x.other + x.cur))
    agg =  agg_.perc[:10]
    agg.reset_index(drop=1).plot()
#     dct[cur_vers] = agg
    aggs.append(agg.to_frame().assign(vers=cur_vers).reset_index(drop=0))
#     break

# %%
stacked = (
    pd.concat(aggs)
    .reset_index(drop=0)
    .rename(columns={"index": "day"})
    .assign(
        percent=lambda x: x.perc.mul(100).round(1),
        cur_vers=lambda x: (x.vers == "69.0.2").map(
            {True: "current", False: "previous"}
        ),
        date=lambda x: x[c.submission_date].astype(str)
    )
)

# %%
stacked[:3]

# %%
base = A.Chart(data=stacked).encode(
    x='day',
    y='perc',
    color='cur_vers',
    detail='vers',
    tooltip=['date', 'day', 'vers', 'percent']
).mark_point()

(base + base.mark_line()).interactive()

# %%

# %%
DataFrame(dct)

# %%
agg

# %%

# %%
wn_.groupby([c.app_version]).n.sum()
