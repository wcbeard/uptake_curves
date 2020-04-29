project_dir="/Users/wbeard/repos/uptake_curves"
export PYTHONPATH="$project_dir:$PYTHONPATH"
TABLE='wbeard.uptake_plot_data'
HTML_DIR='reports/html_prod/'

# TABLE='wbeard.uptake_plot_data'
# HTML_DIR='reports/html_test/'

echo "=> rendering"
python uptake/plot/embed_html.py \
    --sub_date=None \
    --plot_table=$TABLE \
    --project_id='moz-fx-data-bq-data-science' \
    --cache=True \
    --creds_loc=None \
    --html_dir=$HTML_DIR