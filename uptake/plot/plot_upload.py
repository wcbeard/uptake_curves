import datetime as dt
from textwrap import dedent

import fire  # type: ignore
import pandas as pd  # type: ignore

import uptake.bq_utils as bq
import uptake.data.release_dates as rd
import uptake.plot.uptake_plots as up


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

There are 2 date filters for getting the plot upload date. First, it only
pulls the most recent 11 months' worth of data for all channels. Then
there is a channel specific date filter to only use most recent `n` months
of data for a channel-specific `n`.
"""


def process_raw_channel_counts(dfc, prod_details, beta_dates):
    # Release
    pd_rls = prod_details.query(
        "date > '2019' & category in ('major', 'stability')"
    ).pipe(lambda x: x[~x["release_label"].str.endswith("esr")])

    dfcr = dfc.query("chan == 'release'").copy()
    dfcr = up.combine_uptake_dates_release(dfcr, pd_rls, vers_col="dvers")

    # Beta
    beta_dates = rd.get_beta_release_dates(
        min_build_date="2019", min_pd_date="2019-01-01"
    )
    dfcb = dfc.query("chan == 'beta'").copy()
    dfcb = (
        up.combine_uptake_dates_release(dfcb, beta_dates, vers_col="dvers")
        .drop("vers", axis=1)
        .rename(columns={"dvers": "vers"})
    )

    # DevEdition
    dev_dates = rd.pull_bh_data_dev(min_build_date="2019")
    dfc_dev = dfc.query("chan == 'aurora'").copy()
    dfc_dev = (
        up.combine_uptake_dates_release(dfc_dev, dev_dates, vers_col="dvers")
        .drop("vers", axis=1)
        .rename(columns={"dvers": "vers"})
    )

    # Nightly
    dfcn = (
        dfc.query("chan == 'nightly'")
        .copy()
        .assign(build_day=lambda x: x.bid.str[:8])
        .assign(rls_date=lambda x: pd.to_datetime(x.build_day))
        .drop("vers", axis=1)
        .rename(columns={"build_day": "vers"})
    )
    return dfcr, dfcb, dfc_dev, dfcn


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
    dfr,
    dfb,
    dfdev,
    dfn,
    channels_months_ago=(7, 3, 3),
    sub_date: dt.datetime = None,
):
    """
    channels_months_ago: tuple of (release, beta, nightly) number of months to
    pull

    DevEdition just uses the Beta time period.
    """

    def get_min_date(months_ago):
        return (sub_date - pd.Timedelta(days=30 * months_ago)).strftime(
            bq.SUB_DATE
        )

    r_min_date, b_min_date, n_min_date = map(get_min_date, channels_months_ago)
    df_plottable = (
        pd.concat(
            [
                format_channel_data(
                    dfr, min_date=r_min_date, channel="release"
                ),
                format_channel_data(dfb, min_date=b_min_date, channel="beta"),
                format_channel_data(
                    dfdev, min_date=b_min_date, channel="aurora"
                ),
                format_channel_data(
                    dfn, min_date=n_min_date, channel="nightly"
                ),
            ],
            sort=False,
        )
        .assign(
            days_post_pub=lambda x: x.days_post_pub.fillna(float("nan")),
            is_major=lambda x: x.is_major.fillna(False),
            RC=lambda x: x.RC.fillna(False),
        )
        .drop(["nth_recent_release"], axis=1)
    )
    return df_plottable


def main(
    sub_date=None,
    dest_table="analysis.wbeard_uptake_plot_test",
    src_table="analysis.wbeard_uptake_vers",
    project_id="moz-fx-data-derived-datasets",
    src_project_id="moz-fx-data-derived-datasets",
    cache=False,
    creds_loc=None,
    ret_df="all",
    dest_table_exists=True,
):
    """
    sub_date: 'YYYY-mm-dd' or None. If None, then use today.
    Pull 11 months' worth of data from the uptake summaries
    table.
    ret_df: 'all' to return all rows (including rows already)
        uploaded, 'upload' for those about to be uploaded, None
        to return nothing.
    """
    date = pd.to_datetime(sub_date) if sub_date else dt.datetime.today()
    months_ago11 = bq.to_sql_date(
        pd.to_datetime(date) - pd.Timedelta(days=11 * 30)
    )
    prod_details = rd.read_product_details_all()
    beta_dates = rd.get_beta_release_dates(
        min_build_date="2019", min_pd_date="2019-01-01"
    )
    creds = bq.get_creds(creds_loc=creds_loc)
    bq_read = bq.mk_bq_reader(creds_loc=creds_loc, cache=cache)

    def gen_src_query():
        src_loc = bq.BqLocation.from_dataset_table(
            src_table, project_id=src_project_id
        )
        sql = f"""select * from {src_loc.sql}
        where submission_date >= '{months_ago11}'
            and submission_date <= '{bq.to_sql_date(date)}'
        """
        return dedent(sql)

    src_sql = gen_src_query()
    dfc = bq_read(src_sql)

    dfr, dfb, dfdev, dfn = process_raw_channel_counts(
        dfc, prod_details, beta_dates
    )
    df_plottable = format_all_channels_data(
        dfr, dfb, dfdev, dfn, channels_months_ago=(7, 3, 3), sub_date=date
    )

    show_dates = df_plottable.submission_date.map(bq.to_sql_date)
    print(
        f"""Data formatted for dates '{show_dates.min()}' - '{(
            show_dates.max())}' (before filtering)"""
    )

    # Find dates that have already been uploaded. Filter out rows in
    # df_plottable with these dates.
    bq_loc_dest = bq.BqLocation.from_dataset_table(dest_table, project_id)
    q = (
        f"""
        select distinct submission_date as date
        from {bq_loc_dest.sql}
        where submission_date >= '{bq.to_sql_date(
            df_plottable.submission_date.min()
        )}'
        """
    )

    if dest_table_exists:
        distinct_existing_dates = bq_read(q).date.dt.tz_localize(None)
        df_plottable_to_upload = df_plottable.pipe(
            lambda x: x[~x.submission_date.isin(distinct_existing_dates)]
        )
    else:
        df_plottable_to_upload = df_plottable

    print(
        f"""Uploading data for dates:\n{(
            df_plottable_to_upload.submission_date
            .drop_duplicates()
            .sort_values()
            .map(bq.to_sql_date)
            .tolist()
        )}"""
    )

    df_plottable_to_upload.to_gbq(
        dest_table, project_id=project_id, credentials=creds, if_exists="append"
    )
    if ret_df is None:
        return
    elif ret_df == "all":
        return df_plottable
    elif ret_df == "upload":
        return df_plottable_to_upload
    else:
        raise ValueError(f"{ret_df} not one of {{None, 'all', 'upload'}}")


if __name__ == "__main__":
    fire.Fire(main)
