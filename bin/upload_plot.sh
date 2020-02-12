#!/bin/bash
set -eo pipefail
IFS=$'\n\t'

proj="$( dirname "${BASH_SOURCE[0]}" )/.."
project_dir="$(cd "$proj"; pwd)"
echo "Using project_dir: $project_dir"

export PYTHONPATH="$project_dir:$PYTHONPATH"
cd $project_dir

python uptake/upload_bq.py \
    --table_name='wbeard_uptake_vers' \
    --sub_date_end=1 \
    --cache=True

python uptake/plot/plot_upload.py \
    --sub_date=None \
    --dest_table="analysis.wbeard_uptake_plot_test" \
    --src_table="analysis.wbeard_uptake_vers" \
    --project_id="moz-fx-data-derived-datasets" \
    --cache=False \
    --creds_loc=None \
    --ret_df=None
