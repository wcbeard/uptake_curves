#!/bin/bash
set -exo pipefail
IFS=$'\n\t'

INSTANCE="wbeard-uptake"
PROJECT="moz-fx-data-bq-data-science"
ZONE="us-east1-b"
GCP_OUT_CHANNEL_DIR="/home/wbeard/repos/uptake_curves/reports/channel_html"
SCP_DEST=$1
LOG_LOC="/home/wbeard/repos/uptake_curves/logs/log- $(date)"

function gcstart {
    #alias gcssh='gcloud beta compute --project "moz-fx-dev-sguha-rwasm" ssh --zone "us-west1-b" "sguha-datascience"'
    gcloud beta compute instances \
        start $INSTANCE \
        --project=$PROJECT \
        --zone=$ZONE
}

function gcssh {
    gcloud beta compute \
        ssh "$INSTANCE" \
        --project $PROJECT \
        --zone=$ZONE \
        --command "cd /home/wbeard/repos/uptake_curves ;
         rm -rf logfile ; rm -r "$GCP_OUT_CHANNEL_DIR/*" ; bash bin/upload_plot.sh 2>&1 |
         tee > '$LOG_LOC'"
}

function gcscp {
    gcloud compute scp \
    "$INSTANCE:$GCP_OUT_CHANNEL_DIR/*" \
    "$SCP_DEST" \
    --recurse \
    --project "$PROJECT" \
    --zone $ZONE
}

function gcstop {
    gcloud beta compute instances \
        stop "$INSTANCE" \
        --project=$PROJECT \
        --zone $ZONE

}


# gcloud beta compute --project "moz-fx-data-bq-data-science" ssh --zone "us-east1-b" "wbeard-uptake"

gcstart
sleep 10
gcssh
gcscp
gcstop

# sleep 10
# gcloud beta compute --project "moz-fx-dev-sguha-rwasm" ssh "instance-1"  --command " cd /home/sguha/missioncontrol-v2 ; rm -rf logfile; sh complete.runner.sh 2>&1 | tee logfile" --zone=us-central1-c
# rc=$?;
# rc=0
# if [[ -z "$gcstop" ]]; then
#    gcloud beta compute instances --project "moz-fx-dev-sguha-rwasm" stop "instance-1"   --zone=us-central1-c
#  else
#      echo "Not stopping instance-1"
# fi
