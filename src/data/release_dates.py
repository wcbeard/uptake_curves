import re

import buildhub_utils as bh


def pull_bh_data_beta(min_build_date):
    beta_docs = bh.pull_build_id_docs(
        min_build_day=min_build_date, channel="beta"
    )
    return bh.version2df(beta_docs, keep_rc=False, keep_release=True).assign(
        chan="beta"
    )


def pull_bh_data_rls(min_build_date):
    major_re = re.compile(r"^(\d+)\.\d+$")

    def major(v):
        m = major_re.findall(v)
        if not m:
            return None
        [maj] = m
        return int(maj)

    rls_docs = bh.pull_build_id_docs(
        min_build_day=min_build_date, channel="release"
    )
    df = (
        bh.version2df(
            rls_docs, major_version=None, keep_rc=False, keep_release=True
        )
        .assign(chan="release", major=lambda x: x.disp_vers.map(major))
        .assign(is_major=lambda x: x.major.notnull())
    )
    return df
