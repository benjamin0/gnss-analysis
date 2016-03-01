#!/usr/bin/env bash
# Copyright (C) 2016 Swift Navigation Inc.
# Contact: Ben Segal <ben@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

# This script will download multiband CORS data, decompress it, and upload to S3.
# Required:
# crx2rnx: http://terras.gsi.go.jp/ja/crx2rnx.html
# awscli: `pip install awscli`

# Any subsequent commands which fail will cause the shell script to exit immediately
set -e

USAGE='Usage: ./convert_sitl_data.sh -r "cebr" -b "vill" -d "049" -h "23" -m <"00"|"15"|"30"|"45"> -y "2016"'
# Find base and rover stations here: ftp://cddis.gsfc.nasa.gov/pub/gnss/data/highrate/

# If no args, show the help.
if [ $# -eq 0 ]
  then
    echo $USAGE
    exit 1
fi

# Parse input arguments.
while getopts ":r:b:d:h:m:y:" opt; do
  case $opt in
    r)
      ROVER=$OPTARG
      ;;
    b)
      BASE=$OPTARG
      ;;
    d)
      DAY=$OPTARG
      ;;
    h)
      HOUR_NUM=$OPTARG
      ;;
    m)
      MIN=$OPTARG
      ;;
    y)
      YEAR=$OPTARG
      ;;
    *)
      echo $USAGE
      exit 1
      ;;
  esac
done

# Get path to sitl_data folder.
DATA_DIR=$PWD/../../tests/test_data/sitl_data
BUCKET_PREFIX=estimation-sitl-data/multiband-datasets

# Convert hour number to a letter for CORS naming convention.
LETTERS=abcdefghijklmnopqrstuvwx
HOUR_LETTER=${LETTERS:${HOUR_NUM#0}:1}
YEAR_DIGITS=${YEAR:2:2}

# Go to the sitl_data folder.
pushd $DATA_DIR > /dev/null

# Make folder and file names according to CORS naming convention.
FOLDER_NAME="rover_"$ROVER"_base_"$BASE"_"$YEAR"_"$DAY"_"$HOUR_NUM"_"$MIN/
ROVER_FILE=$ROVER$DAY$HOUR_LETTER$MIN".16d"
NAV_FILE=$ROVER$DAY$HOUR_LETTER$MIN".16n"
BASE_FILE=$BASE$DAY$HOUR_LETTER$MIN".16d"

echo "Downloading: "$ROVER_FILE
echo "Downloading: "$NAV_FILE
echo "Downloading: "$BASE_FILE

# Make new folder to hold dataset, but don't fail if it already exists.
mkdir -p $FOLDER_NAME

# Actually download rover and base observation files and navigation file.
wget ftp://cddis.gsfc.nasa.gov/pub/gnss/data/highrate/$YEAR/$DAY/"$YEAR_DIGITS"d/$HOUR_NUM/$ROVER_FILE.Z -P $FOLDER_NAME
wget ftp://cddis.gsfc.nasa.gov/pub/gnss/data/highrate/$YEAR/$DAY/"$YEAR_DIGITS"n/$HOUR_NUM/$NAV_FILE.Z -P $FOLDER_NAME
wget ftp://cddis.gsfc.nasa.gov/pub/gnss/data/highrate/$YEAR/$DAY/"$YEAR_DIGITS"d/$HOUR_NUM/$BASE_FILE.Z -P $FOLDER_NAME

# Unzip files.
gzip -d -f $FOLDER_NAME/*.Z

# Convert from compressed RINEX to regular RINEX files.
crx2rnx -f $FOLDER_NAME/$ROVER_FILE
crx2rnx -f $FOLDER_NAME/$BASE_FILE

# Upload data to S3 bucket.
aws s3 sync $FOLDER_NAME s3://$BUCKET_PREFIX/$FOLDER_NAME

# Go back to tools folder.
popd > /dev/null
