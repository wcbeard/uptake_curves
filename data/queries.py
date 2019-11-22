winq = """
SELECT
  c.submission_date
  , c.app_version
  , count(*) as n
from `telemetry.clients_daily` c
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
from `telemetry.clients_daily` c
where (
--        (submission_date between '2019-07-09' and '2019-07-20') OR
--        (submission_date between '2019-09-01' and '2019-09-14') OR
--        (submission_date between '2019-09-17' and '2019-09-30') OR
       (submission_date between '2019-10-01' and '2019-10-07')
       )
      and sample_id = 1
      and os in ('Windows_NT', 'Darwin', 'Linux')
      and c.channel in ('release', 'beta', 'nightly', 'esr')
      and c.app_name = 'Firefox'
group by 1, 2, 3, 4, 5, 6
"""
