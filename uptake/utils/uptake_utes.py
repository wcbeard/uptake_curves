from typing import List

from pandas.util.testing import assert_frame_equal  # type: ignore


def maj_os(dvers):
    vers = str(dvers).split(".")[0]
    try:
        return int(vers)
    except ValueError:
        return


def max_x_based_on_y(df, x, y, groupby: List[str], transform=True):
    """
    groupby `groupby`, then get argmax of metric x, indexed by y
    """

    def f(gdf):
        return gdf[y].loc[gdf[x].idxmax()]

    y2x_df = df[groupby + [x, y]].groupby(groupby).apply(f)

    if not transform:
        return y2x_df
    merged = df.merge(y2x_df.reset_index(drop=0), how="left", on=groupby)
    assert_frame_equal(merged.iloc[:, :-1], df)
    return merged.iloc[:, -1]
