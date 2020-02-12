import datetime as dt
import json
from pathlib import Path

import altair as A  # type: ignore
import fire  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from uptake import proj_loc
from uptake.plot.plot_upload import to_sql_date
from uptake.plot import uptake_plots as up

# import uptake.plot.uptake_plots as up
import uptake.bq_utils as bq

html_template = """
<!DOCTYPE html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@4"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head>

<body>
  <div id="win"></div>
  <div id="mac"></div>
  <div id="linux"></div>

  <script>
    const win_spec = {win_spec};
    const mac_spec = {mac_spec};
    const linux_spec = {linux_spec};

    vegaEmbed("#win", win_spec)
        // result.view provides access to the Vega View API
      .then(result => console.log(result))
      .catch(console.warn);

    vegaEmbed("#mac", mac_spec)
        // result.view provides access to the Vega View API
      .then(result => console.log(result))
      .catch(console.warn);

    vegaEmbed("#linux", linux_spec)
        // result.view provides access to the Vega View API
      .then(result => console.log(result))
      .catch(console.warn);

  </script>
</body>
"""

json_base = proj_loc / "reports/channel_html"


def convert_np(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def render_channel(win, mac, linux, channel, out_dir):
    html = html_template.format(
        win_spec=json.dumps(win.to_dict(), default=convert_np),
        mac_spec=json.dumps(mac.to_dict(), default=convert_np),
        linux_spec=json.dumps(linux.to_dict(), default=convert_np),
    )
    out = out_dir / f"{channel}.html"
    if not out_dir.exists():
        out_dir.mkdir()
    with open(out, "w") as fp:
        fp.write(html)


templ_sql_rank = """with base as (
select *
from `{plot_table}` u
where channel = '{channel}'
    and submission_date < '{max_sub_date}'
)

, unq_vers as (
select
  os
  , vers
  , channel
  , rank() over (partition by os, channel order by min(b.build_ids) desc)
    as nth_recent_release
from base b
where not b.old_build
group by 1, 2, 3
)

, builds as (
select
  b.*
  , nth_recent_release
from base b
left join unq_vers using (os, vers, channel)
)


select *
from builds
where nth_recent_release is not null
    and nth_recent_release <= {n_releases}
order by os, channel, nth_recent_release
"""


def rm_tz(s):
    return s.dt.tz_localize(None)


def download_channel_plot(
    channel,
    sub_date: dt.date,
    n_releases,
    bq_read,
    plot_table="analysis.wbeard_uptake_plot_test",
):

    q = templ_sql_rank.format(
        plot_table=plot_table,
        channel=channel,
        n_releases=n_releases,
        max_sub_date=to_sql_date(sub_date),
    )
    return bq_read(q).assign(
        submission_date=lambda x: x.submission_date.pipe(rm_tz),
        vers_min_date_above_npct=lambda x: x.vers_min_date_above_npct.pipe(
            rm_tz
        ),
    )


def dl_render_channel(
    channel,
    sub_date: dt.date,
    bq_read,
    plot_table="analysis.wbeard_uptake_plot_test",
):
    channel_n_releases = dict(release=20, beta=30, nightly=40)
    df = download_channel_plot(
        channel,
        sub_date,
        n_releases=channel_n_releases[channel],
        bq_read=bq_read,
        plot_table=plot_table,
    )
    od = up.generate_channel_plot(
        df, A, min_date="2019-06-01", channel="release", separate=True
    )

    render_channel(
        win=od["Windows_NT"],
        mac=od["Darwin"],
        linux=od["Linux"],
        channel=channel,
        out_dir=json_base / to_sql_date(sub_date),
    )
    return df


def main(
    sub_date=None,
    plot_table="analysis.wbeard_uptake_plot_test",
    # project_id="moz-fx-data-derived-datasets",
    cache=True,
    creds_loc=None,
    html_dir=None,
):
    if sub_date is None:
        sub_date = dt.date.today()
    else:
        sub_date = pd.to_datetime(sub_date).date()
    bq_read = bq.mk_bq_reader(cache=cache)
    html_dir = json_base if html_dir is None else Path(html_dir)

    dl_render_channel(
        channel="release",
        sub_date=sub_date,
        bq_read=bq_read,
        plot_table=plot_table,
    )

    dl_render_channel(
        channel="beta",
        sub_date=sub_date,
        bq_read=bq_read,
        plot_table=plot_table,
    )

    dl_render_channel(
        channel="nightly",
        sub_date=sub_date,
        bq_read=bq_read,
        plot_table=plot_table,
    )
    return sub_date


if __name__ == "__main__":
    fire.Fire(main)
