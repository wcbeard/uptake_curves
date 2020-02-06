import pandas as pd
import uptake.bq_utils as bq
from pytest import raises


def test_check_sub_date_format():
    bq.check_sub_date_format(["2019-01-01"])

    with raises(ValueError):
        bq.check_sub_date_format(["01-01-2019"])

    with raises(ValueError):
        bq.check_sub_date_format(pd.to_datetime(["2019-01-01", "2019-12-01"]))
