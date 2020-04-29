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


## Tests
* `bin/test_upload.sh`
* `bin/test_embed.sh`

# GCP
## To fire it up
along lines of `gcp.sh`:

```sh
INSTANCE="wbeard-uptake"
PROJECT="moz-fx-data-bq-data-science"
ZONE="us-east1-b"

gcloud beta compute instances start "wbeard-uptake" --project="moz-fx-data-bq-data-science" --zone="us-east1-b"

gcloud beta compute \
        ssh "$INSTANCE" \
        --project $PROJECT \
        --zone=$ZONE

cd /home/wbeard/repos/uptake_curves

        --command "cd /home/wbeard/repos/uptake_curves ;
         rm -rf logfile ; rm -r "$GCP_OUT_CHANNEL_DIR/*" ; bash bin/upload_plot.sh 2>&1 |
         tee logfile"
```

## Through web interface
* Go to [link](https://console.cloud.google.com/cloud-resource-manager?project=&folder=&organizationId=442341870013)
    * Hamburger -> Compute engine -> VM instances
    * select wbeard-uptake-curve
    * click instance-1 -> start

# Workflows
## On update pipeline

