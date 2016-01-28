# Copyright (C) 2015 Swift Navigation Inc.
#
# This source is subject to the license found in the file 'LICENSE' which must
# be be distributed together with this source. All other rights reserved.
#
# THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

"""
simulate.py

Contains a set of utility functions that run through piksi log
files and simulate the chip state which can be used to perform
analysis and iterate on the algorithm.
"""

import copy
import logging
import numpy as np
import pandas as pd
from functools import partial

import sbp.navigation as nav
import sbp.observation as ob

from gnss_analysis import ephemeris, log_utils
from gnss_analysis.io import sbp_utils


def simulate_from_log(log, initial_state=None):
  """
  Iterates through a log file and emits a dictionary
  of DataFrames holding the current state of observations
  for each rover observation.

  Parameters
  ----------
  log : sbp.base_logger.LogIterator
    An sbp log iterator that should yield msg, data pairs.
    NOTE!!! JSONLogIterator incorrectly iterates, so you
    actually need to pass in JSONLogIterator().next() to
    this method.
  initial_state : (optional) dict
    An optional dictionary containing the state at the
    beginning of logging.

  Returns
  -------
  states : generator
    A generator which yields current estimates of state.
    Each state is keyed by source ('rover', 'base', 'ephemeris')
    and then indexed by satellite id.  For example,
    state['rover'][16] would contain a DataFrame holding
    the rover stations observations of satellite 16.

    A state is only emitted for every rover observation.
  """
  _processors = {ob.MsgObs: sbp_utils.update_observation,
                 ob.MsgObsDepA: sbp_utils.update_observation,
                 ob.MsgEphemeris: sbp_utils.update_ephemeris,
                 ob.MsgEphemerisDepA: sbp_utils.update_ephemeris,
                 ob.MsgEphemerisDepB: sbp_utils.update_ephemeris,
                 nav.MsgPosECEF: partial(sbp_utils.update_position,
                                         suffix='spp_ecef'),
                 nav.MsgPosLLH: partial(sbp_utils.update_position,
                                        suffix='spp_llh'),
                 nav.MsgBaselineNED: partial(sbp_utils.update_position,
                                             suffix='rtk_ned'),
                 nav.MsgBaselineECEF: partial(sbp_utils.update_position,
                                              suffix='rtk_ecef'),
                 nav.MsgGPSTime: sbp_utils.update_gps_time}

  # use either the initial state provided, or a blank set
  # of observations.
  state = initial_state or {'rover': pd.DataFrame(),
                            'base': pd.DataFrame(),
                            'ephemeris': pd.DataFrame(), }

  for msg, data in log_utils.complete_messages_only(log):
    if type(msg) in _processors:
      # TODO: process other messages
      state = _processors[type(msg)](state, msg, data)
      if sbp_utils.is_rover_observation(msg) and not state['rover'].empty:
        # yield a copy so we don't accidentally modify things.
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug("%d, %d" % (msg.header.t.wn, msg.header.t.tow))
        if 'ephemeris' in state:
          state_copy = copy.deepcopy(state)
          # add a time stamp to the ephemeris so we can keep track of which
          # ephemerides were known at which time.
          state_copy['ephemeris']['time'] = state['rover']['time'].values[0]
          yield state_copy
    else:
      logging.debug("No processor available for message type %s"
                    % type(msg))
