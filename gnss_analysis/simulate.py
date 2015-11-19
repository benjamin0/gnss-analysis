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

import time
import logging
import datetime
import numpy as np
import pandas as pd

import sbp.observation as ob

from sbp.utils import exclude_fields
from swiftnav.gpstime import gpst_components2datetime

import gnss_analysis.constants as c


def time_from_message(wn, tow):
  """
  Takes a message with a time field stored as week number and time
  of week and converts it to a TimeStamp.
  """
  # compute the time from the week number and time of week
  dtime = gpst_components2datetime(wn, tow / c.MSEC_TO_SECONDS)
  return dtime


def get_source(msg):
  """
  Determines if a message originated at the rover or base station.
  """
  # if the source field is 0 the source of observations is
  # from the base, otherwise  the rover.
  return 'rover' if msg.sender else 'base'


def get_sid(msg):
  """
  Infer the satellite id from an sbp message (or observation).
  This looks first for an sid attribute, then a prn attribute.
  """
  # does the msg have an sid attribute?
  if hasattr(msg, 'sid'):
    # the sbp interface changed and in some cases the
    # satellite id is held in msg.sid.sat.
    if hasattr(msg.sid, 'sat'):
      return msg.sid.sat
    else:
      return msg.sid
  # does the msg have a prn attribute?
  elif hasattr(msg, 'prn'):
    return msg.prn
  else:
    raise AttributeError("msg does not contain a satellite id")


def observation_to_dataframe(msg, data):
  """
  Convert an observation message to a dataframe which
  is indexed by the msg source and satellite id.

  Parameters
  ----------
  msg : ob.MsgObs or ob.MsgObsDepA
    An sbp ephemeris message.
  data :
    The data corresponding to the msg.

  Returns
  -------
  df : pd.DataFrame
    A DataFrame holding all the observations from various
    satellites.  The DataFrame is indexed by source and
    satellite, ('rover', 16) for example.  The DataFrame
    holds pseudorange, carrier phase
  """
  # make sure the message is an observation message
  assert isinstance(msg, (ob.MsgObs, ob.MsgObsDepA))
  # determine the source
  source = get_source(msg)

  # determine the time from week number and time of week
  dtime = time_from_message(msg.header.t.wn, msg.header.t.tow)
  # converts to seconds from epoch.  This keeps all the data
  # in the resulting DataFrame floats which makes the
  # subsequent update steps a lot faster (since we can avoid
  # the overhead of datetime64 logic).
  timestamp = time.mktime(dtime.timetuple())

  def extract_observations(obs):
      # Convert pseudorange, carrier phase to SI units.
      v = {'P': obs.P / c.CM_TO_M,
           'L': obs.L.i + obs.L.f / c.Q32_WIDTH,
           'cn0': obs.cn0,
           'lock': obs.lock,
           'host_time': data['timestamp'],
           'host_offset': data['delta'],
           'timestamp': timestamp,
           }
      return v

  # Compute the indices of each of the observations
  source_sid_pairs = [(source, get_sid(o)) for o in msg.obs]
  # assemble into a MultiIndex
  idx = pd.MultiIndex.from_tuples(source_sid_pairs,
                                  names=['source', 'sid'])
  # Combine into a dataframe.
  df = pd.DataFrame([extract_observations(o) for o in msg.obs],
                    index=idx)
  return df


def ephemeris_to_dataframe(msg, data):
  """
  Converts an ephermesis log message to a pandas Dataframe
  that is indexed by the source and satellite id.

  Parameters
  ----------
  msg : ob.MsgEphemeris, ob.MsgEphemerisDepA ob.MsgEphemerisDepB
    An sbp ephemeris message.
  data :
    The data corresponding to the msg.

  Returns
  -------
  eph : pd.DataFrame
    A DataFrame indexed by the source and satellite id, ('rover', 16)
    for example.  The columns then hold the ephemeris parameters.
  """
  assert isinstance(msg, (ob.MsgEphemeris,
                          ob.MsgEphemerisDepA,
                          ob.MsgEphemerisDepB))
  # determine if the ephemeris was from the rover or base
  # determine the satellite id.
  # if the message is healthy and valid we emit the corresponding DataFrame
  if msg.healthy == 1 and msg.valid == 1:
    source = get_source(msg)
    sid = get_sid(msg)
    msg = exclude_fields(msg)
    # make sure the satellite id attribute isn't propagated
    [msg.pop(x, None) for x in ['sid', 'prn']]
    # create the MultiIndex and DataFrame
    idx = pd.MultiIndex.from_tuples([(source, sid)],
                                    names=['source', 'sid'])
    eph = pd.DataFrame(msg, index=idx)
    return eph


def tdcp_doppler(old, new):
  """
  Compute the doppler by using the time difference of the
  carrier phase (TDCP).

  Parameters:
  -----------
  old : pd.DataFrame
    A DataFrame holding a previous set of carrier phase
    observations along with corresponding timestamps
  new : pd.DataFrame
    A DataFrame holding new carrier phase observations
    along with corresponding time stamps

  Returns:
  --------
  doppler : pd.Series
    A Series holding the doppler estimates.

  See also: libswiftnav/src/track.c:tdcp_doppler
  """
  # delta time in seconds
  dt = new['timestamp'] - old['timestamp']
  # make sure dt is non-negative.  Sometimes this
  # is zero ?!  In which case should we set the doppler
  # to NaN?  Or use a previously available doppler?
  assert np.all(dt.values >= 0.)
  # compute the rate of change of the carrier phase
  doppler = (new.L - old.L) / dt
  return doppler


def add_doppler(state, updates):
  """
  Computes doppler information from the difference between
  a current state estimate and any updates.  The result
  is stored as a new column 'doppler' in the update DataFrame.

  Parameters
  ----------
  state : pd.DataFrame
    A DataFrame which holds the current estimate of state before
    any updates have been applied.
  updates : pd.DataFrame
    A DataFrame which holds any updates resulting from a new
    log message.

  Returns
  -------
  updates : pd.DataFrame
    A data frame with an additional column holding the doppler
    information if lock information existed.  Otherwise it
    returns an unmodified updates DataFrame.
  """
  # make sure the two DataFrames have already been aligned.
  assert state.index.equals(updates.index)
  # if both the current state and the update contain
  # lock counts, we can compute the doppler shift and
  # do so only for cases where the lock counts haven't changed.
  if 'lock' in state and 'lock' in updates:
    use = state['lock'] == updates['lock']
    if np.any(use):
      # if we lose a lock, should we set doppler to nan, or
      # keep the old value?
      updates['doppler'] = tdcp_doppler(state.ix[use],
                                        updates.ix[use])
  return updates


def initial_state():
  return None


def update_state(state, updates):
  """
  Takes a previous state and an update to the state and
  combines the two.  This also computes any derived quantities
  that require differencing the two states (such as doppler).
  """
  # if no updates, return the original state
  if updates is None:
    return state
  # if there is no previous state return the updates.
  if state is None:
    return updates
  # doing align upfront avoid's doing alignment twice
  updates, state = updates.align(state)
  # infer doppler information and add it to the updates
  updates = add_doppler(state, updates)
  # update the state (or initialize it)

  # pd.DataFrame.combine_first(updates, state) would
  # work here as well, but it is extremely slow.  This
  # version is not nearly as robust but is an order
  # of magnitude faster.
  updates[updates.isnull()] = state
  return updates


def simulate_from_log(log):
  """
  Iterates through a log file and emits a pandas DataFrame
  that holds the current observation state.

  Parameters
  ----------
  log : sbp.base_logger.BaseLogger
    An sbp log iterator that should yield msg, data pairs.

  Returns
  -------
  states : generator
    A generator which yields pandas DataFrames with the
    current estimate of state.  Each DataFrame is indexed
    by source and sid, ('rover', 16) for example, and contains
    the most recent observations and ephemeris information for
    each of the satellites.
  """
  _processors = {ob.MsgObs: observation_to_dataframe,
                 ob.MsgObsDepA: observation_to_dataframe,
                 ob.MsgEphemeris: ephemeris_to_dataframe,
                 ob.MsgEphemerisDepA: ephemeris_to_dataframe,
                 ob.MsgEphemerisDepB: ephemeris_to_dataframe}

  state = initial_state()
  # what's up with iterating over log.next()?
  for msg, data in log.next():
    if type(msg) in _processors:
      updates = _processors[type(msg)](msg, data)
      # TODO: take care of integrity checks
      # TODO: process other messages
      # TODO: remove out-dated information?
      state = update_state(state, updates)
      if state is not None:
        # yield a copy so we don't accidentally modify things.
        yield state.copy()
    else:
      logging.debug("No processor available for message type %s"
                    % type(msg))
