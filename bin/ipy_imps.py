
import subprocess
import tempfile
import time

import bq_utils as bq
import numpy as np
import pandas as pd
import toolz.curried as z
# from data.upload_bq import *
import upload_bq as ub
# from data import crud
# from data.bq_utils import BqLocation, check_table_exists
# from data.crud import mk_bq_reader, mk_query_func, mk_query_func_async
from pandas import Series

query_func = bq.mk_query_func()
bq_loc = bq.BqLocation("wbeard_uptake_vers")
pd.options.display.min_rows = 30

# df = pd.read_csv("data/samp/samp.csv")

lap = z.compose(list, map)
lange = z.compose(list, range)
lilter = z.compose(list, filter)
lip = z.compose(list, zip)

sub_date_start = "2019-12-01"
sub_date_end = "2019-12-10"
dr = Series(pd.date_range(sub_date_start, sub_date_end, freq='D'))
# existing_dates = ub.check_dates_exists('2019-12-01', '2019-12-05', bq_loc)


"""
import sys
sys.path.insert(0, "/Users/wbeard/repos/uptake_curves/src")
%load_ext autoreload
%autoreload 2
from bin.ipy_imps import *

"""
