import datetime as dt

import pandas as pd  # type: ignore

import uptake.plot.uptake_plots as up
import uptake.data.release_dates as rd
import uptake.bq_utils as bq

"""
Module to format raw data counts uploaded to BQ via `upload_bq.py`,
so that it can be easily plotted. The results will be uploaded elsewhere
to BQ.

not null
- is_major: for release
- RC: beta
- dvers: nightly

- what's `vers_min_sday_npct`?
    - first day a os/chan/version had more than 1% of DAU
- what's `nth_recent_release`?
"""


def process_raw_channel_counts(dfc, prod_details, beta_dates):
    pd_rls = prod_details.query(
        "date > '2019' & category in ('major', 'stability')"
    ).pipe(lambda x: x[~x["release_label"].str.endswith("esr")])

    dfcr = dfc.query("chan == 'release'").copy()
    dfcr = up.combine_uptake_dates_release(dfcr, pd_rls, vers_col="dvers")

    beta_dates = rd.get_beta_release_dates(
        min_build_date="2019", min_pd_date="2019-01-01"
    )
    dfcb = dfc.query("chan == 'beta'").copy()
    dfcb = (
        up.combine_uptake_dates_release(dfcb, beta_dates, vers_col="dvers")
        .drop("vers", axis=1)
        .rename(columns={"dvers": "vers"})
    )

    dfcn = (
        dfc.query("chan == 'nightly'")
        .copy()
        .assign(build_day=lambda x: x.bid.str[:8])
        .assign(rls_date=lambda x: pd.to_datetime(x.build_day))
        .drop("vers", axis=1)
        .rename(columns={"build_day": "vers"})
    )
    return dfcr, dfcb, dfcn


def format_channel_data(
    df, channel="release", min_date="2019-10-01", disp_days=None
):
    """
    Convert data pulled from summary table into plottable format,
    with most of the data necessary for plotting in the
    One row for each combination of "vers", "submission_date", "os". The only
    fields that should vary significantly are
    - `build_ids`
    - `nth_recent_release`
    Fields that are usually null:
    - For nightly, `dvers` is like 70.0a1, vers is like '20190801'.
        - so `dvers` is only non-null for nightly
    - rc is only non-null for beta
    - is_major is only non-null for release
    """
    # channel_release_disp_days = dict(release=30, beta=10, nightly=7)
    # passed to generate_channel_plot -> up.os_plot_base_release()
    # max_days_post_pub = disp_days or channel_release_disp_days[channel]

    oses = []
    print(f"{channel}; min_date={min_date}")
    for os in ("Windows_NT", "Darwin", "Linux"):
        osdf = df.query("os == @os")
        pdf = (
            up.format_os_df_plot(osdf, pub_date_col="rls_date", channel=channel)
            .query(f"submission_date > '{min_date}'")
            .assign(os=os)
            # .query("nth_recent_release == nth_recent_release")
        )
        oses.append(pdf)
    return pd.concat(oses).assign(channel=channel)


def format_all_channels_data(
    dfr, dfb, dfn, channels_months_ago=(7, 3, 3), sub_date: str = None
):
    def get_min_date(months_ago):
        date = pd.to_datetime(sub_date) if sub_date else dt.datetime.today()
        return (date - pd.Timedelta(days=30 * months_ago)).strftime(bq.SUB_DATE)

    r_min_date, b_min_date, n_min_date = map(get_min_date, channels_months_ago)
    df_plottable = pd.concat(
        [
            format_channel_data(dfr, min_date=r_min_date, channel="release"),
            format_channel_data(dfb, min_date=b_min_date, channel="beta"),
            format_channel_data(dfn, min_date=n_min_date, channel="nightly"),
        ],
        sort=False,
    )
    return df_plottable


def main(sub_date=None, cache=False):
    prod_details = rd.read_product_details_all()
    beta_dates = rd.get_beta_release_dates(
        min_build_date="2019", min_pd_date="2019-01-01"
    )
    bq_read = bq.mk_bq_reader(creds_loc=None, cache=cache)
    dfc = bq_read("select * from analysis.wbeard_uptake_vers")
    dfr, dfb, dfn = process_raw_channel_counts(dfc, prod_details, beta_dates)
    res = format_all_channels_data(
        dfr, dfb, dfn, channels_months_ago=(7, 3, 3), sub_date=sub_date
    )
    return res
