import datetime as dt
from distutils.dir_util import copy_tree
import json
from pathlib import Path
import subprocess
from typing import Tuple

import altair as A  # type: ignore
import fire  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from uptake import proj_loc

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
  <h1>{channel_header}</h1>
  <h2>{sub_date}</h2>
  <em> Generated {datetime} </em>
  <div>
    Go to:
    <a href="release.html">Release</a>
    <a href="beta.html">Beta</a>
    <a href="nightly.html">Nightly</a>
  </div>

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


def render_channel(win, mac, linux, channel, base_dir, sub_date: str):
    html = html_template.format(
        channel_header=channel.capitalize(),
        sub_date=sub_date,
        win_spec=json.dumps(win.to_dict(), default=convert_np),
        mac_spec=json.dumps(mac.to_dict(), default=convert_np),
        linux_spec=json.dumps(linux.to_dict(), default=convert_np),
        datetime=subprocess.check_output("date").strip(),
    )
    out = base_dir / f"{channel}.html"
    if not base_dir.exists():
        base_dir.mkdir()
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
    and nth_recent_release <= {n_versions}
order by os, channel, nth_recent_release
"""


def rm_tz(s):
    return s.dt.tz_localize(None)


def download_channel_plot(
    channel,
    sub_date: dt.date,
    n_versions,
    bq_read,
    plot_table="analysis.wbeard_uptake_plot_test",
):

    q = templ_sql_rank.format(
        plot_table=plot_table,
        channel=channel,
        n_versions=n_versions,
        max_sub_date=bq.to_sql_date(sub_date),
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
    n_versions,
    plot_table="analysis.wbeard_uptake_plot_test",
    base_dir=json_base,
):
    df = download_channel_plot(
        channel,
        sub_date,
        n_versions=n_versions,
        bq_read=bq_read,
        plot_table=plot_table,
    )
    od = up.generate_channel_plot(
        df, A, min_date="2019-06-01", channel="release", separate=True
    )

    out_dir = base_dir / bq.to_sql_date(sub_date)
    render_channel(
        win=od["Windows_NT"],
        mac=od["Darwin"],
        linux=od["Linux"],
        channel=channel,
        base_dir=out_dir,
        sub_date=bq.to_sql_date(sub_date),
    )
    return out_dir


def main(
    sub_date=None,
    plot_table="analysis.wbeard_uptake_plot_test",
    # project_id="moz-fx-data-derived-datasets",
    cache=True,
    creds_loc=None,
    html_dir=None,
    release_beta_nightly_n_versions: Tuple[int, int, int] = (20, 30, 40),
):
    """
    release_beta_nightly_n_versions: Number of recent releases to pull for
    release, beta, and nightly channels, respectively.
    """
    if sub_date is None:
        sub_date = dt.date.today()
    else:
        sub_date = pd.to_datetime(sub_date).date()
    bq_read = bq.mk_bq_reader(cache=cache)
    html_dir = json_base if html_dir is None else Path(html_dir)
    channels_n_versions = iter(release_beta_nightly_n_versions)

    print(f"html_dir: {html_dir}")
    dl_render_channel(
        channel="release",
        sub_date=sub_date,
        bq_read=bq_read,
        n_versions=next(channels_n_versions),
        plot_table=plot_table,
        base_dir=html_dir,
    )

    dl_render_channel(
        channel="beta",
        sub_date=sub_date,
        bq_read=bq_read,
        n_versions=next(channels_n_versions),
        plot_table=plot_table,
        base_dir=html_dir,
    )

    out_dir = dl_render_channel(
        channel="nightly",
        sub_date=sub_date,
        bq_read=bq_read,
        n_versions=next(channels_n_versions),
        plot_table=plot_table,
        base_dir=html_dir,
    )

    if sub_date == dt.date.today():
        today_dir = str(html_dir / "today")
        print(f"Copying from {out_dir} to {today_dir}")
        copy_tree(out_dir, today_dir)
    return str(out_dir)


if __name__ == "__main__":
    fire.Fire(main)
