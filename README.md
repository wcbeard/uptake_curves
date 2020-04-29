uptake_curves
==============================

Dashboard to compare update curves for new versions

# High level pipeline
* hala (via crontab) starts a gcp instance
* this pulls summary stats from clients_daily, uploads to an intermediate table under `moz-fx-data-bq-data-science`
* another routine pulls the summary stats, transforms to more 'plottable' data, uploads to 2nd intermediate table
* another routine pulls from this table, generates an altair plot, embeds in an html template
* hala scp's the html file to its own data directory

https://metrics.mozilla.com/protected/wbeard/uptake_curves/today/release.html



# Tables
Data is written to 

* raw summary date: `moz-fx-data-bq-data-science.wbeard.uptake_version_counts`
* more directly plottable data: `moz-fx-data-bq-data-science.wbeard.uptake_plot_data`

Data is plotted from `embed_html.py` by directly downloading from the latter
table.




# Hala
## code
- in `/home/wbeard/repos/uptake_curves`
- using master branch
- script that calls it is `/home/wbeard/repos/uptake_curves/bin/gcp_prod.sh`
    - similar to `gcp.sh`
    - this calls `bin/upload_plot.sh` from within a spun up GCP VM
    - scp output to `/home/wbeard/wbeard-prot/uptake_curves/today/`

## upload_plot.sh
runs
- `uptake/upload_bq.py`
- `uptake/plot/plot_upload.py`
- `uptake/plot/embed_html.py`




# Workflows
## On update pipeline

