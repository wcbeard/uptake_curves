project_dir="/Users/wbeard/repos/uptake_curves"
export PYTHONPATH="$project_dir:$PYTHONPATH"

# echo "=> Updating summary table"
# python uptake/upload_bq.py \
#     --table_name='uptake_version_counts_test' \
#     --sub_date_start='2019-04-28' \
#     --project_id="moz-fx-data-bq-data-science" \
#     --add_schema=True \
#     --check_dates=False \
#     --cache=True

echo "=> Updating plot summary table"
python uptake/plot/plot_upload.py \
    --sub_date=None \
    --dest_table="wbeard.uptake_plot_data_test" \
    --src_table="wbeard.uptake_version_counts_test" \
    --project_id="moz-fx-data-bq-data-science" \
    --src_project_id="moz-fx-data-bq-data-science" \
    --cache=True \
    --creds_loc=None \
    --ret_df=None \
    
    # --dest_table_exists=False \

# --sub_date_end='2020-03-01' \
    # --sub_date_start='2019-03-01' \
    # --sub_date_end='2019-04-01' \

    # --drop_first=True \

    # --sub_date_start='2019-04-02' \
    # --sub_date_end=1 \
