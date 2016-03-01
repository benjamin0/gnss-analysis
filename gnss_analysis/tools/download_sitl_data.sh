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

# This script will download the collection of SITL datasets from S3.
# Required:
# awscli: `pip install awscli`

# Any subsequent commands which fail will cause the shell script to exit immediately
set -e

# Get path to sitl_data folder.
DATA_DIR=$PWD/../../tests/test_data/sitl_data
BUCKET_PREFIX=estimation-sitl-data/multiband-datasets

# Go to the sitl_data folder.
pushd $DATA_DIR > /dev/null

# Download all data in the bucket/prefix.
aws s3 sync s3://$BUCKET_PREFIX $DATA_DIR

# Go back to tools folder.
popd > /dev/null
