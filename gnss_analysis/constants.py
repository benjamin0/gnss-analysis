#!/usr/bin/env python
# Copyright (C) 2015 Swift Navigation Inc.
# Contact: Ian Horn <ian@swiftnav.com>
#          Bhaskar Mookerji <mookerji@swiftnav.com>
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""
constants.py

Holds constants from libswiftnav as well as any that are gnss_analysis specific.
"""

from swiftnav import constants as sc

# First we import any constants defined in libswiftnav
R2D = sc.R2D_
D2R = sc.D2R_

GPS_PI = sc.GPS_PI_
GPS_L1_HZ = sc.GPS_L1_HZ_
GPS_OMEGAE_DOT = sc.GPS_OMEGAE_DOT_
GPS_GM = sc.GPS_GM_
GPS_C = sc.GPS_C_
GPS_F = sc.GPS_F_
GPS_C_NO_VAC = sc.GPS_C_NO_VAC_
GPS_L1_LAMBDA = sc.GPS_L1_LAMBDA_
GPS_L1_LAMBDA_NO_VAC = sc.GPS_L1_LAMBDA_NO_VAC_
GPS_NOMINAL_RANGE = sc.GPS_NOMINAL_RANGE_
GPS_CA_CHIPPING_RATE = sc.GPS_CA_CHIPPING_RATE_

DEFAULT_PHASE_VAR_TEST = sc.DEFAULT_PHASE_VAR_TEST_
DEFAULT_CODE_VAR_TEST = sc.DEFAULT_CODE_VAR_TEST_
DEFAULT_PHASE_VAR_KF = sc.DEFAULT_PHASE_VAR_KF_
DEFAULT_CODE_VAR_KF = sc.DEFAULT_CODE_VAR_KF_
DEFAULT_AMB_DRIFT_VAR = sc.DEFAULT_AMB_DRIFT_VAR_
DEFAULT_AMB_INIT_VAR = sc.DEFAULT_AMB_INIT_VAR_
DEFAULT_NEW_INT_VAR = sc.DEFAULT_NEW_INT_VAR_

MAX_CHANNELS = sc.MAX_CHANNELS_

# Then add additional one

# TODO(Buro): Move these constants below to
# libswiftnav/libswiftnav-python.

# Units conversations (length, time, and 8-bit fractional cycles)
MAX_SATS = 32
MSEC_TO_SECONDS = 1000.
MM_TO_M = 1000.
CM_TO_M = 100.
Q32_WIDTH = 256.

# Constants and settings ported from Piksi firmware

# Solution constants
TIME_MATCH_THRESHOLD = 2e-3
OBS_PROPAGATION_LIMIT = 10e-3
MAX_AGE_OF_DIFFERENTIAL = 1.0
OBS_N_BUFF = 5

# Solution state
SOLN_MODE_LOW_LATENCY = 0
SOLN_MODE_TIME_MATCHED = 1
DGNSS_SOLUTION_MODE = SOLN_MODE_LOW_LATENCY

# RTK filter state
FILTER_FLOAT = 0
FILTER_FIXED = 1
dgnss_filter_state = FILTER_FLOAT

# RTK SHIT
MIN_SATS = 4

# Use Ephemeris from the last four hours
EPHEMERIS_TOL = 3600 * 4

# Constants from libswiftnav (include/libswiftnav/constants.h)
MAX_SATS = 32
WEEK_SECS = 7 * 24 * 60 * 60