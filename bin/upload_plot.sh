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

# echo "=> Updating summary table"
# python uptake/upload_bq.py \
#     --table_name='uptake_version_counts' \
#     --sub_date_end=1 \
#     --cache=True

# echo "=> Updating plot summary table"
# python uptake/plot/plot_upload.py \
#     --sub_date=None \
#     --dest_table="wbeard.uptake_plot_data" \
#     --src_table="wbeard.uptake_version_counts" \
#     --project_id="moz-fx-data-bq-data-science" \
#     --cache=False \
#     --creds_loc=None \
#     --ret_df=None

echo "=> Generating plots"
full_html_output=$(python uptake/plot/embed_html.py \
    --sub_date=None \
    --plot_table="wbeard.uptake_plot_data" \
    --release_beta_nightly_n_versions="12,18,30" \
    --cache=True)

outdir=$(echo "$full_html_output" | tail -n 1)
echo "outdir $outdir"
