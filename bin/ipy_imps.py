
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

# df = pd.read_csv("data/samp/samp.csv")

lap = z.compose(list, map)
lange = z.compose(list, range)
lilter = z.compose(list, filter)
lip = z.compose(list, zip)

sub_date_start = "2019-12-04"
sub_date_end = "2019-12-04"

"""
import sys
sys.path.insert(0, "/Users/wbeard/repos/uptake_curves/src")
%load_ext autoreload
%autoreload 2
from bin.ipy_imps import *

"""
