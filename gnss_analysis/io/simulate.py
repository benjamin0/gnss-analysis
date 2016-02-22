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

Functions for simulating what a receiver would be observing from
observations files (such as RINEX and sbp logs).

These simulate functions were built with the goal of having a standard
way of developing filters or performing analysis in a way that is
indpendent of the input format.  More specifically, each simulate
function is responsible for taking some input format and producing
an iterable of what we call "observation sets".  The observation sets
produced from  RINEX file should be (nearly) identical to those
produced from piksi log files.

An observation set is generated for each epoch.  These self describing
observation_sets typically consist of rover observations (assumed
to be valid for the given epoch), ephemerides (also valid for, but
not necessarily received on, the given epoch), base observations 
(valid on or before the given epoch) and additional information required
to describe the observation set.

Observation Set : An observation set is intended to represent a collection
  of all the information available to a receiver at a given epoch.  This
  information is contained in a dictionary with the following fields:
  
  'rover': A pd.DataFrame that consists of all the rover observations
      valid on the epoch.  This field is always contained in an
      observation set.
  'rover_info': A dictionary of attributes associated with the
      rover observations.  These attributes may include things like
      a flag indicating whether the observations have been adjusted for
      receiver clock error.
  'base': A pd.DataFrame that consists of the most recent base observations.
      These observations are often (but not neccesarily) valid for the
      epoch.  If the base observations were delayed during transmission
      (for example) they may be valid for an epoch previous to the
      current one.
  'base_info': A dictionary of attributes associated with the
      rover observations.  These attributes may include things like
      a flag indicating whether the observations have been adjusted for
      receiver clock error.
  'ephemeris' : A pd.DataFrame that contains all the ephemeris/navigation
      parameters that describe satellites available in rover and
      base observations.  The set of satellites in the ephemeris do
      no neccesarily contain all, nor are they constrained to, the set of
      satellites observed by the rover or base.
  'ephemeris_info': A dictionary of attributes associated with the navigation
      (aka ephemeris) data.  This may include things like the offset
      relative to UTC etc.
  'epoch' : A np.datetime64[ns] representation of the epoch the observation
      set is valid for.
  'attributes': A dictionary containing any additional information
      pertinent to the understanding of the observations in the set.
      Typically this should not contain any information that will change
      between epochs.
      
Observation : A pd.DataFrame that contains all measurements made by
  a reciever.  The DataFrame rows are indexed by a unique satellite id (sid),
  which shoudl consist of a concatenation of the constellation, satellite
  number and band.  Measurements from the same satellite but different
  frequencys are treated as two seperate measurements.  In addition to the
  actual measurement values an observation DataFrame should include a
  'constellation', 'sat' and 'band' column which can be used to perform
  common satellite/constellation/band joins.

  In addition to satellite descriptors, the columns will typically include:
    'carrier_phase', 'raw_pseudorange', 'time', 'signal_strength',
    'lock', 'pseudorange', 'tot'.
"""

import copy
import logging
import numpy as np
import pandas as pd
from functools import partial

import sbp.navigation as nav
import sbp.observation as ob

from gnss_analysis import log_utils
from gnss_analysis.io import sbp_utils, rinex, hdf5


def get_unique_value(x):
  """
  Convenience function that tries to reduce x to a single
  unique value.
  """
  uniq = np.unique(x)
  assert uniq.size == 1
  return uniq[0]


def simulate_from_log(log, initial_observation_set={}):
  """
  Iterates through a log file and emits a dictionary
  of DataFrames each of which holds a set of observations
  for a single epoch.

  Parameters
  ----------
  log : string (or sbp.base_logger.LogIterator)
    The path to an sbp log, or optionally an instance of
    sbp.client.loggers.JSONLogIterator.
  initial_observation_set : (optional) dict
    An optional dictionary containing the observations at the
    epoch immediately before logging starts.  This allows the user
    to bootstrap simulation or set default header information.

  Returns
  -------
  obs_sets : generator
    A generator which yields observation sets for each epoch.
    See simulate.py for a description of observation sets.

    A set of observations is emitted for each observed rover epoch.
  """
  # TODO: We could reformulate the way sbp logs are read to match the
  #   (header, iterator) form used by rinex / hdf5 which would result
  #   in all these simulate functions using the same alignment logic.
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
                                             suffix='rtk_ned',
                                             in_mm=True),
                 nav.MsgBaselineECEF: partial(sbp_utils.update_position,
                                              suffix='rtk_ecef',
                                              in_mm=True)}
                # because the piksi propagates all observations to the epoch
                # we actively ignore the gps time messages that hold the
                # time from the SPP solve since it doesn't correspond to the
                # solution time we would get using the sent observations.

  default_obs_set = {'rover': pd.DataFrame(),
                     'rover_info': {'original_format': 'sbp',
                                    'reciever_offset_applied': True},
                     'base_info': {'original_format': 'sbp',
                                   'reciever_offset_applied': True},
                     'ephemeris_info': {'original_format': 'sbp', }, }
  # use either the initial observation set provided, or a blank set
  # of observations.
  obs_set = default_obs_set
  obs_set.update(initial_observation_set)

  for msg, data in log_utils.complete_messages_only(log):
    if type(msg) in _processors:
      obs_set = _processors[type(msg)](obs_set, msg, data)
      if sbp_utils.is_rover_observation(msg) and not obs_set['rover'].empty:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug("%d, %d" % (msg.header.t.wn, msg.header.t.tow))
        if 'ephemeris' in obs_set:
          # yield a copy so we don't accidentally modify things.
          obs_set_copy = copy.deepcopy(obs_set)
          obs_set_copy['epoch'] = get_unique_value(obs_set['rover']['epoch'])
          obs_set_copy['rover'].drop('epoch', axis=1, inplace=True)
          if 'base' in obs_set_copy:
            obs_set_copy['base'].drop('epoch', axis=1, inplace=True)
          yield obs_set_copy
    else:
      logging.debug("No processor available for message type %s"
                    % type(msg))


def simulate_from_iterators(rover, **kwdargs):
  """
  Creates an iterator of observation sets, one for each epoch
  in the rover observations.
  
  This function is intended to perform the epoch alignment for
  multiple different input formats.
  
  Parameters
  ----------
  rover : (header_info, observation_iterator)
    rover (and all subsequent observations) are expected to be
    a tuple of header_info dictionary and observation iterator.
    The header_info should be a dictionary holding any information
    required to help interpret the observations.  The observation
    iteratator is intended to be an iterator that yields
    pd.DataFrame objects that correspond to observations for a
    particular epoch in chronological order.

  
  """
  rover_info, iter_rover_obs = rover
  # a dictionary of iterators from which we pull observations
  iterators = {k: iter_obs for k, (info, iter_obs) in kwdargs.iteritems()
               if iter_obs is not None}

  obs_set = {'rover_info': rover_info}
  # add all the information dictionaries from other observations
  obs_set.update({'%s_info' % k: info
                  for k, (info, iter_obs) in kwdargs.iteritems()})

  # a dictionary holding all the next observations for each group
  next_obs = {k: v.next() for k, v in iterators.iteritems()}

  def maybe_use_next(group_name, cur_obs_set, update=False):
    # If the next navigation message comes before
    # the current rover observation we update the obs_set
    while np.any(next_obs[group_name]['epoch'] <= cur_obs_set['epoch']):
      # the epoch attribute is stored at the root level of an
      # observaiton set, rather than worry about keeping all observations
      # consistent we drop the epoch variable from the observation
      if 'time' in next_obs[group_name]:
        assert np.all(next_obs[group_name]['time'] <= cur_obs_set['epoch'])

      no_epoch = next_obs[group_name].drop('epoch', axis=1)

      if update and group_name in cur_obs_set:
        # this is used for ephemeris information which typically
        # arrives gradually.  Rather than overwritting the current
        # observations we want to add (update) the new ones.
        old, new = cur_obs_set[group_name].align(no_epoch, 'outer')
        old.update(new)
        cur_obs_set[group_name] = old
      else:
        # for most situations we want to completely overwrite
        # the current observations with the new ones.
        cur_obs_set[group_name] = no_epoch
      # Try and get the next observation.  If there aren't
      # any more we simply break out and continue with the most
      # recent.  It'll be up to the user to realize it is old and
      # decide if it's still usable.
      try:
        next_obs[group_name] = iterators[group_name].next()
      except StopIteration:
        # If a given iterator runs out of items we don't want the
        # entire generator function to stop, we'd like to continue
        # until we run out of rover observations.
        logging.debug("Exhausted all %s observations" % group_name)
        break

  # The goal is to produce a generator that yields an obs
  # for each observed rover epoch.  Here we iterate over
  # rover epoch's and search for the corresponding base and
  # ephemeris messages.
  for rover_obs in iter_rover_obs:
    # Each rover observation should have a unique epoch time,
    # we use that to sync up the rest of the observations.
    # The resulting observaiton set should have only the most
    # recent observations
    obs_set['epoch'] = get_unique_value(rover_obs['epoch'])
    # update the rover obs
    obs_set['rover'] = rover_obs.drop('epoch', axis=1)

    # For all other observations we compare the time they
    # were observed to the current epoch and update accordingly.
    for group in iterators.keys():
      maybe_use_next(group, obs_set, update=group == 'ephemeris')

    # and return a copy of the dict (though not a deep copy since
    # that causes uneccesary overhead.
    yield copy.deepcopy(obs_set)



def simulate_from_rinex(rover, navigation=None, base=None):
  """
  Takes filelike objects for rover, navigation and base files
  and generates observations for each epoch in the rover files
  which contains the most recent ephemeris and base
  observations up to that epoch.
  """
  if navigation is None:
    navigation = rinex.infer_navigation_path(rover)

  return simulate_from_iterators(rover=rinex.read_observation_file(rover),
                                 ephemeris=rinex.read_navigation_file(navigation),
                                 base=rinex.read_observation_file(base))


def simulate_from_hdf5(hdf5_file):
  """
  Creates an iterable of observation sets from an HDF5 file.
  This is done by creating iterators which yield rover, ephemeris
  and base observations by epoch then using simulate_from_iterators
  to perform epoch alignment.
  
  Parameters
  ----------
  hdf5_file : string
    The path to an HDF5 file.
  
  Returns
  -------
  observation_sets : generator
    A generator which yields observation sets, one for each epoch
    stored in the rover group of the HDF5 file.
  """

  with pd.HDFStore(hdf5_file, mode='r') as store:
    # determine all the available groups
    keys = [x.strip('/') for x in store.keys()]
    assert 'rover' in keys
    # this creates a header, iterator tuple for each group
    kwdargs = {k: hdf5.read_group(store, k) for k in keys}
    # We iterate over this (instead of just returning the generator)
    # in order to keep the hdf5 file open.
    for obs_set in simulate_from_iterators(**kwdargs):
      yield obs_set
