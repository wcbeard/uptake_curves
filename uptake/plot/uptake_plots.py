from collections import OrderedDict

import pandas as pd  # type: ignore
from pandas.testing import assert_frame_equal  # type: ignore
import uptake.data.release_dates as rd  # type: ignore
import uptake.utils.uptake_utes as ut


def combine_uptake_dates_release(df, rls_date_df, vers_col="dvers"):
    """
    Add `version` and `date` colums to df to get release dates.
    These release datese will be pulled from `product details`
    and `buildhub` (for beta rc).
    @df: version updake data that contains `vers_col`
    @rls_date_df: df with `version` and `date` columns
        to tell when a version was released.
    """
    rls_date_df = rls_date_df[["version", "date"]]
    v2date = dict(zip(rls_date_df.version, rls_date_df.date))
    df_ret = df.assign(rls_date=lambda df: df[vers_col].map(v2date))
    assert set(df_ret) - set(df) == {"rls_date"}
    return df_ret


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
    return "\n".join(sorted(bids))


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


def get_release_order(df):
    df = df[["vers", "vers_min_date_above_npct", "build_ids"]]
    n_recent_versions = (
        df.query("vers != 'other'")
        .pipe(lambda x: x[x.build_ids.str.len() > 8])
        .groupby(["vers"])
        .vers_min_date_above_npct.min()
        .sort_values(ascending=False)
        .reset_index(drop=0)
        .assign(nth_recent_release=lambda x: range(1, len(x) + 1))
    )
    vers2n = n_recent_versions.set_index("vers").nth_recent_release.to_dict()
    return df.vers.map(vers2n)


#############################
# Plot formatting functions #
#############################
def format_os_df_plot(os_df, pub_date_col="pub_date", channel="release"):
    """
    - needs a version column called `vers`
    - contains data for single os & channel
    - collapse all build_id's into single version
    `vers_min_date_above_npct` column is first submission_date that a
    channel/os/version combination had at least 1% of the channel/os DAU.
    """
    if channel == "nightly":
        vers_df = os_df[["vers", "dvers", "n"]]
    cols_sort = ["vers", "submission_date", "n"]
    cols_all = cols_sort + ["bid", pub_date_col, "chan", "os"]

    os_df = os_df[cols_all]

    latest_version = (
        os_df[["bid", "vers"]]
        .drop_duplicates()
        .sort_values(by=["bid"], ascending=False)
        .vers.iloc[0]
    )

    gb = (
        os_df[cols_all]
        .sort_values(cols_sort, ascending=[True, True, False])
        .assign(
            day_chan_os_sum=lambda x: x.groupby(
                ["submission_date", "chan", "os"]
            ).n.transform("sum")
        )
        .assign(
            build_ids=lambda x: x.bid,
            min_build_id=lambda x: x.bid,
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
                "build_ids": order_bids,
                "min_build_id": "min",
                "n": "sum",
                "day_chan_os_sum": "max",
                "day_chan_os_sum_min": "min",
            }
        )
        .reset_index(drop=0)
        .assign(
            n_pct=lambda x: x.n / x.day_chan_os_sum,
            latest_vers=lambda x: x.vers.eq(latest_version),
            old_build=lambda x: x.build_ids.map(len) <= 8,
        )
        .assign(
            vers_min_date_above_npct=lambda x: x.vers.map(
                get_vers_first_day_above_n_perc_mapping(x, 0.01)
            )
        )
        .assign(
            days_post_pub=lambda x: sub_days_null(
                x.submission_date, x.vers_min_date_above_npct
            ),
            nth_recent_release=lambda x: get_release_order(x),
        )
    )

    if channel == "release":
        pdf = pdf.assign(is_major=lambda x: is_major(x.vers))
    elif channel == "beta":
        pdf = pdf.assign(RC=lambda x: x.vers.map(rd.is_rc))
    elif channel == "nightly":
        pdf = pdf.assign(
            dvers=lambda x: x.vers.map(map_nightly_vers2dvers(vers_df))
        )
    else:
        raise NotImplementedError(f"channel `{channel}` not yet implemented.")

    # Some `days_post_pub` are negative. There are slight off-by-one errors
    # when there's a tiny rollout on day 1 (like 1%), so the entire series
    # is offset. This 'un-offsets' it
    min_days_post_sub = pdf.groupby("vers").days_post_pub.transform("min")
    pdf.loc[min_days_post_sub == -1, "days_post_pub"] += 1

    assert pdf.eval(
        "day_chan_os_sum_min == day_chan_os_sum"
    ).all(), "hopefully single values at level of grouping"
    return pdf


def map_nightly_vers2dvers(vers_df):
    max_dvers = vers_df.groupby(["vers", "dvers"]).n.max().reset_index(drop=0)
    dvers = ut.max_x_based_on_y(
        max_dvers, "n", "dvers", ["vers"], transform=True
    )
    vers2dvers = (
        max_dvers.assign(dvers2=dvers).set_index("vers").dvers2.to_dict()
    )
    return vers2dvers


key = ["os", "build_ids", "vers"]


def get_channel_ordered_versions(df):
    """
    For each os, order all the `n` versions from 1 to n+1,
    where 1 is the most recent.
    Transform this rank-ordering into tidy-format.
    This functionality should be replicated on the SQL side.
    """
    unique_ordered_os_vers = (
        df.query("~old_build")
        .groupby(["os", "vers"])
        .build_ids.min()
        .reset_index(drop=0)
        .sort_values(["os", "build_ids"], ascending=[True, False])[key]
    )
    unique_ordered_os_vers[
        "recent_version_order"
    ] = unique_ordered_os_vers.groupby("os").vers.transform(
        lambda s: range(1, len(s) + 1)
    )
    return unique_ordered_os_vers


def merge_channel_ordered_versions(df):
    unique_ordered_os_vers = get_channel_ordered_versions(df)
    df2 = (
        df
        # .reset_index(drop=1)
        .reset_index(drop=0)
        .merge(
            unique_ordered_os_vers[["os", "vers", "recent_version_order"]],
            on=["os", "vers"],
            how="left",
        )
        .set_index("index")
        .sort_index()
    )
    assert_frame_equal(
        df.reset_index(drop=1),
        df2.drop(["recent_version_order"], axis=1).reset_index(drop=1),
    )
    return df2


######################
# Plotting functions #
######################
def os_plot_base_release(
    df,
    color="vers:O",
    separate=False,
    A=None,
    channel="release",
    max_days_post_pub=14,
):
    """
    A: altair with working version
    """

    if channel in ("release", "beta"):
        shape_field = "is_major" if channel == "release" else "RC"
        major_shape = A.Shape(
            shape_field,
            scale=A.Scale(
                domain=[True, False], range=["square", "triangle-up"]
            ),
        )
        version_opacity = A.Opacity(
            "latest_vers:O", scale=A.Scale(domain=[True, False], range=[1, 0.6])
        )
    elif channel == "nightly":
        major_shape = A.Undefined
        # major_shape = A.Shape('dvers:O')
        version_opacity = A.Undefined
    else:
        raise NotImplementedError(f"channel `{channel}` not yet implemented.")

    # Pretty names
    pdf = df.query(
        f"days_post_pub <= {max_days_post_pub} & days_post_pub > -2"
    ).assign(
        show_n=lambda x: x.nth_recent_release,
        version=lambda x: x.vers,
        week_day=lambda x: x.submission_date.dt.weekday_name.str[:3],
        days_after_release=lambda x: x.days_post_pub,
        release_day=lambda x: x.vers_min_date_above_npct,
    )

    xscale = A.Scale(domain=[-1, pdf.days_post_pub.max() + .5])
    h = (
        A.Chart(pdf)
        .mark_line(strokeDash=[5, 2])
        .encode(
            x=A.X(
                "days_post_pub",
                axis=A.Axis(title="Days after release"),
                scale=xscale,
            ),
            y=A.Y("n_pct", axis=A.Axis(format="%", title="Percent uptake")),
            # strokeOpacity=version_opacity,
            # shape=major_shape,
            color="vers:O",
            tooltip=[
                "version",
                "days_after_release",
                "submission_date",
                "week_day",
                "release_day",
                "build_ids",
            ],
        )
    )
    # encoding issue https://stackoverflow.com/q/59652075/386279
    h.encoding.strokeOpacity = version_opacity
    h.encoding.shape = major_shape

    if channel == "nightly":
        h = h.encode(shape="dvers")

    if separate:
        return h, h.mark_point()
    combined = h + h.mark_point()

    # Add slider
    os = pdf.os.iloc[0]
    n_slider = A.binding_range(
        min=pdf.show_n.min(), max=pdf.show_n.max(), step=1
    )
    slider_selection = A.selection_single(
        bind=n_slider,
        fields=["show_n"],
        name=f"Num_versions_{os}",
        init={"show_n": 10},
    )
    combined = combined.add_selection(slider_selection).transform_filter(
        A.datum.show_n <= slider_selection.show_n
    )

    return combined


###########################
# Per Channel combination #
###########################
def generate_channel_plot_full(
    df, A, min_date="2019-10-01", channel="release", separate=False
):
    channel_release_disp_days = dict(release=30, beta=10, nightly=7)
    max_days_post_pub = channel_release_disp_days[channel]

    od = OrderedDict()
    for os in ("Windows_NT", "Darwin", "Linux"):
        osdf = df.query("os == @os")
        pdf = (
            format_os_df_plot(osdf, pub_date_col="rls_date", channel=channel)
            .query(f"submission_date > '{min_date}'")
            .assign(os=os)
            .query("nth_recent_release == nth_recent_release")
        )
        ch = (
            os_plot_base_release(
                pdf,
                color="vers:O",
                separate=False,
                A=A,
                channel=channel,
                max_days_post_pub=max_days_post_pub,
            )
            .properties(height=500, width=700, title=f"OS = {os}")
            .interactive()
        )
        od[os] = ch
    if separate:
        return od
    return (
        A.concat(*od.values())
        .resolve_scale(color="independent")
        .resolve_axis(y="shared")
    )


def generate_channel_plot(
    df, A, min_date="2019-10-01", channel="release", separate=False
):
    """
    Creates channel plot from data already uploaded to `plot_upload`.
    """
    channel_release_disp_days = dict(release=30, beta=10, nightly=7)
    max_days_post_pub = channel_release_disp_days[channel]

    od = OrderedDict()
    for os in ("Windows_NT", "Darwin", "Linux"):
        pdf = df.query("os == @os")
        ch = (
            os_plot_base_release(
                pdf,
                color="vers:O",
                separate=False,
                A=A,
                channel=channel,
                max_days_post_pub=max_days_post_pub,
            )
            .properties(height=500, width=700, title=f"OS = {os}")
            .interactive()
        )
        od[os] = ch
    if separate:
        return od
    return (
        A.concat(*od.values())
        .resolve_scale(color="independent")
        .resolve_axis(y="shared")
    )
