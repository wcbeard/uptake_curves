import datetime as dt
from typing import Union

import bq_utils as bq
import fire
import pandas as pd
# from queries import pull_min as pull_min
import queries
from bq_utils import BqLocation, drop_table, mk_bq_reader, upload
from google.cloud import bigquery  # noqa
from google.oauth2 import service_account  # noqa


def download_version_counts(bq_read, sub_date_start, sub_date_end=None):
    q = queries.pull_min(day1=sub_date_start, day_end=sub_date_end)
    res = bq_read(q)
    return res


def check_dates_exists(
    sub_date_start, sub_date_end, bq_loc: bq.BqLocation, creds_loc=None
):
    """
    Before uploading new data, check that the submission_dates in a range
    defined by `sub_date_start` and `sub_date_end` haven't already
    been loaded into the table. Raise a value error if they've been uploaded
    already.
    """
    bq.check_sub_date_format([sub_date_start, sub_date_end])
    date_range = pd.date_range(sub_date_start, sub_date_end, freq="D")  # noqa

    q = f"""
    select submission_date as date
      , count(*) as n_rows
    from {bq_loc.sql}
    group by 1
    """
    bq_read = bq.mk_bq_reader(creds_loc=creds_loc, cache=False)
    existing_dates = bq_read(q)
    duplicate_dates = existing_dates.query("date in @date_range").reset_index(
        drop=1
    )
    if len(duplicate_dates):
        with pd.option_context("display.min_rows", 30):
            print("Warning: the following dates have been uploaded already:")
            print(duplicate_dates, end="\n\n")
            dates_str = ", ".join(duplicate_dates.date.astype(str))
            raise ValueError(f"Following dates already uploaded: {dates_str}")

    return duplicate_dates


def main(
    table_name,
    sub_date_start,
    sub_date_end: Union[str, int],
    dataset="analysis",
    project_id="moz-fx-data-derived-datasets",
    add_schema=False,
    drop_first=False,
    cache=True,
    check_dates=True,
):
    """
    sub_date_start and sub_date_end should be in `%Y-%m-%d` format. If
    `sub_date_end` is an integer `n`, it pull up until `n` days ago.
    """
    bq_loc = BqLocation(table_name, dataset, project_id=project_id)
    bq_read = mk_bq_reader(cache=cache)
    if isinstance(sub_date_end, int):
        sub_date_end = (
            dt.date.today() - pd.Timedelta(days=sub_date_end)
        ).strftime("%Y-%m-%d")
    if check_dates:
        check_dates_exists(sub_date_start, sub_date_end, bq_loc)

    df = download_version_counts(
        bq_read, sub_date_start, sub_date_end=sub_date_end
    )

    if drop_first:
        drop_table(bq_loc)
    upload(df, bq_loc, add_schema=add_schema)
    return


if __name__ == "__main__":
    fire.Fire(main)
