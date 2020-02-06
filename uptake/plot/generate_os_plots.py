import pathlib
import sys
import datetime as dt

project_dir = pathlib.Path(__file__).absolute().parent.parent
sys.path.insert(0, str(project_dir))

from os.path import join

from altair_saver import save  # type: ignore
import altair.vegalite.v3 as A  # type: ignore
import fire
from joblib import Memory
import pandas as pd  # type: ignore
import uptake.bq_utils as bq
import uptake.data.release_dates as rd
import uptake.plot.uptake_plots as up

mem = Memory(cachedir="cache", verbose=0)


def days_ago(n, as_str=True):
    d = dt.date.today() - pd.Timedelta(days=n)
    if not as_str:
        return d
    return d.strftime("%Y-%m-%d")


def plot_release(df_cts, prod_details, min_date="2019-06-01"):
    pd_rls = prod_details.query(
        "date > '2019' & category in ('major', 'stability')"
    ).pipe(lambda x: x[~x["release_label"].str.endswith("esr")])

    dfcr = df_cts.query("chan == 'release'").copy()
    dfcr = up.combine_uptake_dates_release(dfcr, pd_rls, vers_col="dvers")

    chs_rls = up.generate_channel_plot(dfcr, A, min_date=min_date)
    return chs_rls


def plot_beta(df_cts, min_date="2019-10-01"):
    beta_dates = rd.get_beta_release_dates(
        min_build_date="2019", min_pd_date="2019-01-01"
    )
    dfcb = df_cts.query("chan == 'beta'").copy()
    dfcb = (
        up.combine_uptake_dates_release(dfcb, beta_dates, vers_col="dvers")
        .drop("vers", axis=1)
        .rename(columns={"dvers": "vers"})
    )
    chs_beta = up.generate_channel_plot(
        dfcb, A, min_date=min_date, channel="beta"
    )
    return chs_beta


def plot_nightly(df_cts, min_date="2019-10-01"):
    dfcn = (
        df_cts.query("chan == 'nightly'")
        .copy()
        .assign(build_day=lambda x: x.bid.str[:8])
        .assign(rls_date=lambda x: pd.to_datetime(x.build_day))
        .drop("vers", axis=1)
        .rename(columns={"build_day": "vers"})
    )
    ch_nightly = up.generate_channel_plot(
        dfcn, A, min_date=min_date, channel="nightly"
    )
    return ch_nightly


@mem.cache
def read_uptake():
    bq_read = bq.mk_bq_reader(creds_loc=None, cache=False)
    return bq_read("select * from analysis.wbeard_uptake_vers")


def main(days_ago_release=30 * 6, days_ago_beta=90, days_ago_nightly=60):
    google_drive = "/Users/wbeard/Google Drive/uptake/html"
    df_cts = read_uptake()
    prod_details = rd.read_product_details_all()

    chs_rls = plot_release(
        df_cts, prod_details, min_date=days_ago(days_ago_release)
    )
    chs_beta = plot_beta(df_cts, min_date=days_ago(days_ago_beta))
    ch_nightly = plot_nightly(df_cts, min_date=days_ago(days_ago_nightly))

    with A.data_transformers.enable("default"):
        save(chs_rls, join(google_drive, "os_release.html"), inline=True)
        save(chs_beta, join(google_drive, "os_beta.html"), inline=True)
        save(ch_nightly, join(google_drive, "os_nightly.html"), inline=True)


if __name__ == "__main__":
    fire.Fire(main)
    # main()
    # 1
    # print(project_dir / "reports/figures/os_release.html")
