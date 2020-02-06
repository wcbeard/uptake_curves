import datetime as dt
from typing import Optional, Union

import bq_utils as bq
import fire  # type: ignore
import pandas as pd  # type: ignore

# from queries import pull_min as pull_min
import queries
from bq_utils import BqLocation, drop_table, mk_bq_reader, upload


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


def get_latest_missing_date(bq_loc: bq.BqLocation, creds_loc=None):
    q = f"""
    select max(submission_date) as date
    from {bq_loc.sql}
    """
    bq_read = bq.mk_bq_reader(creds_loc=creds_loc, cache=False)
    max_dates = bq_read(q)
    [max_date] = max_dates.date
    max_missing_date = max_date + pd.Timedelta(days=1)
    return max_missing_date.strftime(bq.SUB_DATE)


def main(
    table_name,
    sub_date_start: Optional[str] = None,
    sub_date_end: Union[str, int] = 1,
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
    if `sub_date_start` is None, look up the most recently missing date
    that has already been uploaded.
    """
    bq_loc = BqLocation(table_name, dataset, project_id=project_id)
    bq_read = mk_bq_reader(cache=cache)
    if sub_date_start is None:
        sub_date_start = get_latest_missing_date(bq_loc)
        print(f"No start date passed. Using {sub_date_start}")

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
