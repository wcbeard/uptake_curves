import datetime as dt
import re

import pandas as pd  # type: ignore

import uptake.data.buildhub_utils as bh  # type: ignore
from requests import get


############
# Buildhub #
############
def pull_bh_data_beta(min_build_date):
    beta_docs = bh.pull_build_id_docs(
        min_build_day=min_build_date, channel="beta"
    )
    return bh.version2df(beta_docs, keep_rc=False, keep_release=True).assign(
        chan="beta"
    )


def pull_bh_data_dev(min_build_date):
    beta_docs = bh.pull_build_id_docs(
        min_build_day=min_build_date, channel="aurora"
    )
    df = (
        bh.version2df(beta_docs, keep_rc=True, keep_release=True)
        .assign(chan="aurora")
        .rename(columns={"pub_date": "date", "disp_vers": "version"})
    )
    return df[["version", "build_id", "date", "chan"]]


def pull_bh_data_rls(min_build_date):
    major_re = re.compile(r"^(\d+)\.\d+$")

    def major(v):
        m = major_re.findall(v)
        if not m:
            return None
        [maj] = m
        return int(maj)

    rls_docs = bh.pull_build_id_docs(
        min_build_day=min_build_date, channel="release"
    )
    df = (
        bh.version2df(
            rls_docs, major_version=None, keep_rc=False, keep_release=True
        )
        .assign(chan="release", major=lambda x: x.disp_vers.map(major))
        .assign(is_major=lambda x: x.major.notnull())
    )
    return df[
        ["disp_vers", "build_id", "pub_date", "chan", "major", "is_major"]
    ]


###################
# Product details #
###################
maj_release_url = (
    "https://product-details.mozilla.org"
    "/1.0/firefox_history_major_releases.json"
)


def read_product_details_all():
    pd_url = "https://product-details.mozilla.org/1.0/firefox.json"
    js = pd.read_json(pd_url)
    df = (
        pd.DataFrame(js.releases.tolist())
        .assign(release_label=js.index.tolist())
        .assign(date=lambda x: pd.to_datetime(x.date))
    )
    return df


def read_product_details_maj(min_yr=2018):
    """
    Be careful, some versions are non-integer (like '14.0.1').
    The `min_yr` filter should prevent errors.
    """
    maj_json = get(maj_release_url).json()
    df = (
        pd.DataFrame(maj_json.items(), columns=["vers_float", "date"])
        .assign(date=lambda x: pd.to_datetime(x.date))
        .query(f"date > '{min_yr}'")
        .assign(vers_float=lambda x: x.vers_float.astype(float))
        .reset_index(drop=1)
        .assign(vers=lambda x: x.vers_float.astype(int))
    )
    assert (df.vers == df.vers_float).all(), "Some versions aren't integers"
    return df.drop("vers_float", axis=1)


def get_recent_release_from_product_details() -> int:
    """
    Query product-details.mozilla.org/ to get most
    recent major release. (e.g., 71)
    """
    rls_prod_details_json = get(maj_release_url).json()
    rls_prod_details = pd.Series(rls_prod_details_json).sort_values(
        ascending=True
    )
    [(cur_rls_vers, _date)] = rls_prod_details[-1:].iteritems()
    cur_rls_maj, *_v = cur_rls_vers.split(".")
    return int(cur_rls_maj)


def get_min_build_date(days_ago=90):
    min_build_datetime = dt.datetime.today() - dt.timedelta(days=days_ago)
    return min_build_datetime.strftime("%Y%m%d")


############
# Combined #
############
def sortable_beta_vers(v):
    "70.0b7 -> 70.0b07"
    if "b" not in v:
        return v
    maj, min = v.split("b")
    return f"{maj}b{int(min):02}"


def is_rc(v):
    """
    Is pattern like 69.0, rather than 69.0b3
    """
    return "b" not in v


def get_beta_release_dates(
    min_build_date="2019", min_pd_date="2019-01-01"
) -> pd.DataFrame:
    """
    Stitch together product details and buildhub
    (for rc builds).
    - read_product_details_all()
    - this will only pull rc builds from vers=71.0 onward
    """
    bh_beta_rc = (
        pull_bh_data_beta(min_build_date=min_build_date)
        .rename(columns={"pub_date": "date", "disp_vers": "version"})
        .assign(rc=lambda x: x.version.map(is_rc), src="buildhub")
        .query("rc")[["date", "version", "src"]]
        .sort_values(["date"], ascending=True)
        .drop_duplicates(["version"], keep="first")
    )
    prod_details_all = read_product_details_all()
    prod_details_beta = prod_details_all.query(
        f"category == 'dev' & date > '{min_pd_date}'"
    )[["date", "version"]].assign(src="product-details")

    beta_release_dates = (
        prod_details_beta.append(bh_beta_rc, ignore_index=False)
        .assign(
            maj_vers=lambda x: x.version.map(lambda x: int(x.split(".")[0]))
        )
        .sort_values(["date"], ascending=True)
        .reset_index(drop=1)
        # Round down datetimes to nearest date
        .assign(date=lambda x: pd.to_datetime(x.date.dt.date))
        # .astype(str)
    )
    return beta_release_dates[["date", "version", "src", "maj_vers"]]


def latest_n_release_beta(beta_release_dates, sub_date, n_releases: int = 1):
    """
    Given dataframe with beta release dates and a given
    submission date (can be from the past till today), return the
    `n_releases` beta versions that were released most recently.
    beta_release_dates: df[['date', 'version', 'src', 'maj_vers']]
    """
    beta_release_dates = beta_release_dates[
        ["date", "version", "src", "maj_vers"]
    ]
    latest = (
        beta_release_dates
        # Don't want versions released in the future
        .query("date < @sub_date")
        .sort_values(["date"], ascending=True)
        .iloc[-n_releases:]
    )

    return latest
