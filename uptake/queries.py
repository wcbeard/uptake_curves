winq = """
SELECT
  c.submission_date
  , c.app_version
  , count(*) as n
from `moz-fx-data-derived-datasets.telemetry.clients_daily` c
where (
       (submission_date between '2019-07-09' and '2019-07-20') OR
       (submission_date between '2019-09-01' and '2019-09-14') OR
       (submission_date between '2019-09-17' and '2019-09-30') OR
       (submission_date between '2019-10-01' and '2019-10-07')
       )
      and sample_id = 1
      and os = 'Windows_NT'
      and c.channel = 'release'
      and c.app_name = 'Firefox'
      and c.app_build_id >= '20190601'
group by 1, 2
-- order by 1
"""


allq = """
SELECT
  c.submission_date
  , c.channel as chan
  , c.os
  , c.app_version as vers
  , case when app_build_id < '{min_build_id}' then '{min_build_id}'
      else app_build_id end as bid
  , c.app_display_version as dvers
  , count(*) as n
from `moz-fx-data-derived-datasets.telemetry.clients_daily` c
where (
--        (submission_date between '2019-07-09' and '2019-07-20') OR
--        (submission_date between '2019-09-01' and '2019-09-14') OR
--        (submission_date between '2019-09-17' and '2019-09-30') OR
       (submission_date between '{min_sub_date}' and '2019-10-07')
       )
      and sample_id = 1
      and os in ('Windows_NT', 'Darwin', 'Linux')
      and c.channel in ('release', 'beta', 'nightly', 'esr')
      and c.app_name = 'Firefox'
group by 1, 2, 3, 4, 5, 6
"""


def pull_min(day1, day_end=None):
    day_end = day_end or day1
    allq_min = """
    CREATE TEMP FUNCTION build_date_lag_days(chan string, os string) AS (
      case when chan = 'esr' then (365 * 3)
           when (chan, os) = ('release', 'Linux') then (365 * 2)
           when chan = 'beta' then 100
           else 60
           end
    );

    CREATE TEMP FUNCTION min_build_id(chan string, os string, sub_date date)
      AS (
      format_date('%Y%m%d', date_sub(sub_date,
                  interval build_date_lag_days(chan, os) day))
    );

    -- TODO: delete this?
    CREATE TEMP FUNCTION min_build_id_offset(
            chan string, os string, sub_date date, offset int64)
      AS (
      format_date('%Y%m%d', date_sub(sub_date,
                  interval build_date_lag_days(chan, os) + offset day))
    );


    with aggs_base as (
    SELECT
      c.submission_date
      , c.channel as chan
      , c.os
      , c.app_version as vers
      , if(
           app_build_id < min_build_id(channel, os, submission_date),
           min_build_id(channel, os, submission_date),
           app_build_id) as bid
      , c.app_display_version as dvers
      , count(*) as n
    from `moz-fx-data-derived-datasets.telemetry.clients_daily` c
    where (
           (submission_date between '{day1}' and '{day_end}')
          )
          and sample_id = 1
          and os in ('Windows_NT', 'Darwin', 'Linux')
          and c.channel in ('release', 'beta', 'nightly', 'esr')
          and c.app_name = 'Firefox'
    group by 1, 2, 3, 4, 5, 6
    )

    , chan_os_aggs as (
    select submission_date
      , chan
      , os
      , sum(n) as n
    from aggs_base
    group by 1, 2, 3
    )

    , aggs as (
    select
      a.submission_date
      , a.chan
      , a.os
      , a.vers
      , a.bid
      , a.dvers
      , a.n
      , n.n as os_chan_n
      , a.n / n.n * 100 as perc
      -- reduce ("red") these dimensions if they're too small
      -- as a percentage
      , if(a.n / n.n > .005, dvers, 'other') as red_dvers
      , if(a.n / n.n > .005, vers, 'other') as red_vers
      , if(a.n / n.n > .005, bid,
           min_build_id_offset(a.chan, a.os, a.submission_date, 1)) as red_bid
    from aggs_base a
    join chan_os_aggs n
      on a.submission_date = n.submission_date
      and a.chan = n.chan
      and a.os = n.os
    )

    , reduced_agg as (
    select
      submission_date
      , chan
      , os
      , red_dvers as dvers
      , red_vers as vers
      , red_bid as bid
      , sum(n) as n
    from aggs
    group by 1, 2, 3, 4, 5, 6
    )

    select *
    from reduced_agg
    """.format(
        day1=day1, day_end=day_end
    )

    return allq_min
