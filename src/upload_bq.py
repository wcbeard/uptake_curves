import bq_utils as bq
import fire
from bq_utils import BqLocation, drop_table, mk_bq_reader, upload
from google.cloud import bigquery  # noqa
from google.oauth2 import service_account  # noqa
from queries import pull_min as pull_min


def download_version_counts(bq_read, sub_date_start, sub_date_end=None):
    q = pull_min(day1=sub_date_start, day_end=sub_date_end)
    res = bq_read(q)
    return res


def check_dates_exists(sub_date_start, sub_date_end):
    bq.check_sub_date_format([sub_date_start, sub_date_end])
    pd.date_range(sub_date_start, sub_date_end, freq='D')


def main(
    table_name,
    sub_date_start,
    sub_date_end=None,
    dataset="analysis",
    project_id="moz-fx-data-derived-datasets",
    add_schema=False,
    drop_first=False,
):
    bq_read = mk_bq_reader(cache=True)
    df = download_version_counts(
        bq_read, sub_date_start, sub_date_end=sub_date_end
    )

    loc = BqLocation(table_name, dataset)

    if drop_first:
        drop_table(loc)
    upload(df, loc, add_schema=add_schema)
    return


if __name__ == "__main__":
    fire.Fire(main)
