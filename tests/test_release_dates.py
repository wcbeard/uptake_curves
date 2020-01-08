from src.data.release_dates import sortable_beta_vers  # type: ignore


def test_sortable_beta_vers():
    assert sortable_beta_vers("70.0b4") == "70.0b04"
    assert sortable_beta_vers("70.0") == "70.0"
    assert sortable_beta_vers("67.0b14") == "67.0b14"
