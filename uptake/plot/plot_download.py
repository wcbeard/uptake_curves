sql_rank = """with base as (
select *
from `analysis.wbeard_uptake_plot_test` u
)

, unq_vers as (
select
  os
  , vers
  , channel
  , rank() over (partition by os, channel order by min(b.build_ids) desc)
    as rank
from base b
where not b.old_build
group by 1, 2, 3
)

, builds as (
select
  b.*
  , rank
from base b
left join unq_vers using (os, vers, channel)
)


select *
from builds
--where rank is not null
order by os, channel, rank
"""
