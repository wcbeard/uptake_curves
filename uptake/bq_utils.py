import os
import re
import subprocess
import tempfile
from functools import partial
from os.path import abspath, exists, expanduser
from typing import Iterable

from google.cloud import bigquery  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google import auth  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas import Series

SUB_DATE = "%Y-%m-%d"


def default_proj(proj):
    env_proj = os.environ.get("BQ_PROJ")
    if env_proj:
        print(f"Using project {env_proj}")
        return env_proj
    return proj


class BqLocation:
    def __init__(
        self,
        table,
        dataset="wbeard",
        project_id=default_proj("moz-fx-data-derived-datasets"),
    ):
        self.table = table
        self.dataset = dataset
        self.project_id = project_id

    @property
    def sql(self):
        return f"`{self.project_id}`.{self.dataset}.{self.table}"

    @property
    def cli(self):
        return "{}:{}.{}".format(self.project_id, self.dataset, self.table)

    @property
    def no_proj(self):
        return "{}.{}".format(self.dataset, self.table)

    @property
    def sql_dataset(self):
        return f"`{self.project_id}`.{self.dataset}"

    def __repr__(self):
        return f"BqLocation[{self.cli}]"


def get_creds(creds_loc=None):
    if creds_loc:
        creds_loc = abspath(expanduser(creds_loc))
        creds = service_account.Credentials.from_service_account_file(creds_loc)
    else:
        creds, _proj_id = auth.default()
    return creds


def mk_bq_reader(creds_loc=None, cache=False):
    """
    Returns function that takes a BQ sql query and
    returns a pandas dataframe
    """
    creds = get_creds(creds_loc=creds_loc)

    bq_read = partial(
        pd.read_gbq,
        project_id=default_proj("moz-fx-data-derived-datasets"),
        credentials=creds,
        dialect="standard",
    )
    if cache:
        fn = cache_reader(bq_read)
        loc = os.path.abspath(fn.store_backend.location)
        print("Caching sql results to {}".format(loc))
        return fn
    return bq_read


def cache_reader(bq_read):
    from joblib import Memory  # noqa

    if not exists("cache"):
        os.mkdir("cache")
    mem = Memory(cachedir="cache", verbose=0)

    @mem.cache
    def bq_read_cache(*a, **k):
        return bq_read(*a, **k)

    return bq_read_cache


def mk_query_func(creds_loc=None):
    """
    This function will block until the job is done...and
    take a while if a lot of queries are repeatedly made.
    """
    creds = get_creds(creds_loc=creds_loc)

    client = bigquery.Client(
        project=default_proj(creds.project_id), credentials=creds
    )

    def blocking_query(*a, **k):
        job = client.query(*a, **k)
        for i in job:
            break
        assert job.done(), "Uh oh, job not done??"
        return job

    return blocking_query


def mk_query_func_async(creds_loc=None):
    creds = get_creds(creds_loc=creds_loc)
    client = bigquery.Client(
        project=default_proj(creds.project_id), credentials=creds
    )
    return client.query


def run_command(cmd, success_msg="Success!"):
    """
    @cmd: List[str]
    No idea why this isn't built into python...
    """
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        success = True
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
        success = False

    if success:
        print(success_msg)
        print(output)
    else:
        print("Command Failed")
        print(output)


def drop_table(location: BqLocation):
    cmd = ["bq", "rm", "-f", "-t", location.cli]
    print("running command", cmd)
    run_command(cmd, "Success! Table {} dropped.".format(location.cli))


def upload(df, loc: BqLocation, add_schema=False):
    with tempfile.NamedTemporaryFile(delete=False, mode="w+") as fp:
        df.to_csv(fp, index=False, na_rep="NA")

    print("CSV saved to {}".format(fp.name))

    cmd = [
        "bq",
        "load",
        "--noreplace",
        "--project_id",
        "moz-fx-data-bq-data-science",
        "--source_format",
        "CSV",
        "--skip_leading_rows",
        "1",
        "--null_marker",
        "NA",
        loc.no_proj,
        fp.name,
    ]
    if add_schema:
        schema = get_schema(df, True)
        cmd.append(schema)
    print(cmd)

    success_msg = "Success! Data uploaded to {}".format(loc.no_proj)
    run_command(cmd, success_msg)


def get_schema(df, as_str=False, **override):
    dtype_srs = df.dtypes
    dtype_srs.loc[dtype_srs == "category"] = "STRING"
    dtype_srs.loc[dtype_srs == "float64"] = "FLOAT64"
    dtype_srs.loc[dtype_srs == np.int] = "INT64"
    dtype_srs.loc[dtype_srs == object] = "STRING"
    dtype_srs.loc[dtype_srs == bool] = "BOOL"
    dt_bm = dtype_srs.astype(str).str.contains("datetime64")
    dtype_srs.loc[dt_bm] = "DATE"

    missing_override_keys = set(override) - set(dtype_srs.index)
    if missing_override_keys:
        raise ValueError("Series missing keys {}".format(missing_override_keys))
    dtype_srs.update(Series(override))

    non_strings = dtype_srs.map(type).pipe(lambda x: x[x != str])
    if len(non_strings):
        raise ValueError(
            "Schema values should be strings: {}".format(non_strings)
        )
    if not as_str:
        return dtype_srs
    res = ",".join(["{}:{}".format(c, t) for c, t in dtype_srs.items()])
    return res


def check_sub_date_format(dates: Iterable[str]):
    "Ensure `dates` are strings with date formate `YYYY-dd-mm`"
    date_fmt_re = re.compile(r"\d{4}-\d{2}-\d{2}")
    date_srs = Series(dates)
    if not date_srs.map(type).pipe(set) == {str}:
        raise ValueError("Dates passed aren't strings")
    if not date_srs.map(date_fmt_re.match).astype(bool).all():
        raise ValueError("Date do not follow `YYYY-dd-mm` pattern")
