from functools import partial

import pandas as pd  # type: ignore
from google.oauth2 import service_account  # type: ignore
from fire import Fire  # type: ignore

# creds, default_project_id = auth.default()


def mk_bq_reader(creds_loc):
    """
    Returns function that takes a BQ sql query and
    returns a pandas dataframe
    """
    creds = service_account.Credentials.from_service_account_file(creds_loc)

    bq_read = partial(
        pd.read_gbq,
        project_id="moz-fx-data-bq-data-science",
        credentials=creds,
        dialect="standard",
    )
    return bq_read


def main(creds_loc):
    bq_read = mk_bq_reader(creds_loc=creds_loc)
    res = bq_read(
        """
        select max(submission_date) as date
        from `moz-fx-data-bq-data-science`.wbeard.uptake_version_counts
        --from `moz-fx-data-derived-datasets`.analysis.wbeard_uptake_vers
    """
    )
    print(res)


if __name__ == "__main__":
    Fire(main)
