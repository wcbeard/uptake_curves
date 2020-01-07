import pandas as pd  # type: ignore
import toolz.curried as z  # type: ignore


def combine_uptake_dates_release(df, rls_date_df, vers_col="dvers"):
    """
    @df: version updake data that contains `vers_col`
    @rls_date_df: df with `version` and `date` columns
        to tell when a version was released.
    """
    rls_date_df = rls_date_df[["version", "date"]]
    v2date = dict(zip(rls_date_df.version, rls_date_df.date))
    df["rls_date"] = df[vers_col].map(v2date)
    return df


def is_major(vers: pd.Series):
    "Is major release? As opposed to a dot release."
    return vers.str.count(r"\.").eq(1)


def sub_days_null(s1, s2, fillna=None):
    """
    Subtract dates and `fillna`
    """
    diff = s1 - s2
    not_null = diff.notnull()
    nn_days = diff[not_null].astype("timedelta64[D]").astype(int)
    diff.loc[not_null] = nn_days
    return diff


def order_bids(bids):
    return "\n".join(bids)


def arr_itg(n):
    def f(srs):
        return srs.values[n]

    return f


def get_vers_first_day_above_n_perc_mapping(df, min_pct=0.01):
    df = df[["n_pct", "submission_date", "vers"]]
    return (
        df.query("n_pct > @min_pct")
        .groupby("vers")
        .submission_date.min()
        .to_dict()
    )


def format_os_df_plot(os_df, pub_date_col="pub_date"):
    """
    - contains data for single os & channel
    - collapse all build_id's into single version
    """
    _cs_sort = ["vers", "submission_date", "n"]
    _cs_sort_all = _cs_sort + ["bid", pub_date_col, "chan", "os"]

    os_df = os_df[_cs_sort_all]

    latest_version = (
        os_df[["bid", "vers"]]
        .drop_duplicates()
        .sort_values(by=["bid"], ascending=False)
        .vers.iloc[0]
    )

    gb = (
        os_df[_cs_sort_all]
        .sort_values(_cs_sort, ascending=[True, True, False])
        .assign(
            day_chan_os_sum=lambda x: x.groupby(
                ["submission_date", "chan", "os"]
            ).n.transform("sum")
        )
        .assign(
            bids=lambda x: x.bid,
            vers_pub_date=lambda x: x.groupby(["vers"])[pub_date_col].transform(
                "min"
            ),
            day_chan_os_sum_min=lambda x: x.day_chan_os_sum,
        )
        .groupby(["vers", "submission_date"], sort=False)
    )

    pdf = (
        gb.agg(
            {
                "bids": order_bids,
                "n": "sum",
                "day_chan_os_sum": "max",
                "day_chan_os_sum_min": "min",
            }
        )
        .reset_index(drop=0)
        .assign(
            n_pct=lambda x: x.n / x.day_chan_os_sum,
            latest_vers=lambda x: x.vers.eq(latest_version),
            is_major=lambda x: is_major(x.vers),
        )
        .assign(
            vers_min_sday_npct=lambda x: x.vers.map(
                get_vers_first_day_above_n_perc_mapping(x, 0.01)
            )
        )
        .assign(
            days_post_pub=lambda x: sub_days_null(
                x.submission_date, x.vers_min_sday_npct
            )
        )
    )

    assert pdf.eval(
        "day_chan_os_sum_min == day_chan_os_sum"
    ).all(), "hopefully single values at level of grouping"
    return pdf


def os_plot_base_release(df, color="vers:O", separate=False, A=None):
    """
    A: altair with working version
    """
    version_opacity = A.Opacity(
        "latest_vers:O", scale=A.Scale(domain=[True, False], range=[1, 0.6])
    )
    major_shape = A.Shape(
        "is_major",
        scale=A.Scale(domain=[True, False], range=["square", "triangle-up"]),
    )
    h = (
        A.Chart(df.query("days_post_pub <= 14 & days_post_pub > -2"))
        .mark_line(strokeDash=[5, 2])  # , strokeOpacity=.6
        .encode(
            x=A.X("days_post_pub", axis=A.Axis(title="Days after release")),
            y=A.Y("n_pct", axis=A.Axis(format="%", title="Percent uptake")),
            strokeOpacity=version_opacity,  # 'latest_vers:O',
            color="vers:O",
            shape=major_shape,
            tooltip=[
                "vers",
                "days_post_pub",
                "submission_date",
                "vers_min_sday_npct",
                "bids",
            ],
        )
    )
    if separate:
        return h, h.mark_point()
    return h + h.mark_point()


# For plotting a single version updake
def _mk_rls_plot(
    rls_date,
    df,
    version,
    vers_col="vers",
    os="Windows_NT",
    min_max_pct=10,
    A=None,
):
    rls_date = pd.to_datetime(rls_date)
    beg_win = rls_date - pd.Timedelta(days=3)  # noqa
    end_win = rls_date + pd.Timedelta(days=13)  # noqa
    w = (
        df.query(f"os == '{os}'")
        .query("submission_date >= @beg_win")
        .query("submission_date <= @end_win")
    )
    w2 = (
        w.groupby(["submission_date", vers_col])
        .n.sum()
        .reset_index(drop=0)
        .assign(dayn=lambda x: x.groupby("submission_date").n.transform("sum"))
        .assign(n_perc=lambda x: x.n / x.dayn * 100)
        .assign(
            n_perc_vers=lambda x: x.groupby(vers_col).n_perc.transform("max")
        )
        .query("n_perc_vers > @min_max_pct")
    )
    d1 = {"submission_date": rls_date, "n_perc": 0, vers_col: "release"}
    d2 = z.merge(d1, dict(n_perc=90))
    w2 = w2.append([d1, d2], ignore_index=True)

    h = (
        A.Chart(w2)
        .mark_line()
        .encode(
            x="submission_date",
            y="n_perc",
            color=vers_col,
            tooltip=["n", "n_perc", "submission_date", vers_col],
        )
    ).properties(title=str(version))

    hh = (h + h.mark_point()).interactive()
    return w2, hh
