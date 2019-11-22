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
from boot_utes import (reload, add_path, path, run_magics)
add_path('..', '../src/', '~/repos/myutils/', )

from utils.uptake_curves_imps import *; exec(pu.DFCols_str); exec(pu.qexpr_str); run_magics()
# import utils.en_utils as eu; import data.load_data as ld; exec(eu.sort_dfs_str)

sns.set_style('whitegrid')
A.data_transformers.enable('json', prefix='data/altair-data')
S = Series; D = DataFrame

from big_query import bq_read

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


# %% [markdown]
# # Load

# %%
@mem.cache
def bq_read_cache(*a, **k):
    return bq_read(*a, **k)

winq = '''
select
  c.submission_date
  , c.app_version 
  , count(*) as n
from `telemetry.clients_daily` c
where (
       (submission_date between '2019-07-09' and '2019-07-20') OR
       (submission_date between '2019-09-01' and '2019-09-14') OR
       (submission_date between '2019-09-17' and '2019-09-30') OR
       (submission_date between '2019-10-01' and '2019-10-07')
       )
      and sample_id = 1
      and os = 'Windows_NT'
      and c.channel = 'release'
      and c.app_name = 'Firefox'
      and c.app_build_id >= '20190601'
group by 1, 2
-- order by 1
'''

# %%
wn_ = bq_read_cache(winq).assign(submission_date=lambda x: pd.to_datetime(x.submission_date))

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
