#!/bin/bash
set -eo pipefail
IFS=$'\n\t'

proj="$( dirname "${BASH_SOURCE[0]}" )/.."
project_dir="$(cd "$proj"; pwd)"
echo "Using project_dir: $project_dir"

export PYTHONPATH="$project_dir:$PYTHONPATH"
cd $project_dir

source ~/miniconda3/etc/profile.d/conda.sh
conda activate uptake_curves
export BQ_PROJ="moz-fx-data-bq-data-science"

# Test/Prod
export COUNT_TABLE="uptake_version_counts_test"
export PLOT_TABLE="uptake_plot_data_test"
CACHE="True"


echo "=> Updating summary table"
python uptake/upload_bq.py \
    --table_name=$COUNT_TABLE \
    --sub_date_end=1 \
    --project_id=$BQ_PROJ \
    --cache=True


echo "=> Updating plot summary table"
python uptake/plot/plot_upload.py \
    --sub_date=None \
    --src_table="wbeard.$COUNT_TABLE" \
    --dest_table="wbeard.$PLOT_TABLE" \
    --project_id=$BQ_PROJ \
    --src_project_id=$BQ_PROJ \
    --cache=$CACHE \
    --creds_loc=None \
    --ret_df=None


echo "=> Generating plots"
full_html_output=$(python uptake/plot/embed_html.py \
    --sub_date=None \
    --plot_table="wbeard.$PLOT_TABLE" \
    --project_id=$BQ_PROJ \
    --release_beta_dev_nightly_n_versions="12,18,18,30" \
    --cache=$CACHE)


outdir=$(echo "$full_html_output" | tail -n 1)
echo "outdir $outdir"