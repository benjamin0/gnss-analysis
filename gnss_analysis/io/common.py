import warnings
import numpy as np

from gnss_analysis import time_utils


def tdcp_doppler(old, new):
  """
  Compute the doppler by using the time difference of the
  carrier phase (TDCP).

  Parameters:
  -----------
  old : pd.DataFrame
    A DataFrame holding a previous set of carrier phase
    observations along with corresponding tow
  new : pd.DataFrame
    A DataFrame holding new carrier phase observations
    along with corresponding tow

  Returns:
  --------
  doppler : pd.Series
    A Series holding the doppler estimates.

  See also: libswiftnav/src/track.c:tdcp_doppler
  """
  # TODO: make sure week numbers are the same since we don't
  #  take them into account yet.  That will require doing some
  #  NaN comparisons as well.

  # TODO: This function does not handle receiver clock error,
  #  in particular receiver clock drift (since we're differencing).
  #  In order to fully handle such error we would need to know
  #  if the observations have been propagated to 'time' yet.

  # delta time in seconds
  dt = time_utils.seconds_from_timedelta(new['time'] - old['time'])
  # make sure dt is positive.
  assert not np.any(dt.values <= 0.)
  # compute the rate of change of the carrier phase
  doppler = (new.carrier_phase - old.carrier_phase) / dt
  # mark any computations with differing locks as nan
  invalid = np.mod(new['lock'], 2) == 1
  doppler[invalid] = np.nan
  return doppler


def create_sid(constellation, sat, band):
  """
  The satellite id (sid) is a unique identifier of a satellite signal
  which is created by combining a satellite (which typically consists
  of a constellation identifier followed by satellite number) and the
  band.  For example: 'G12-2' would be the L2 signal from GPS constellation
  12.
  """
  sat = np.asarray(sat).astype('S')
  band = np.asarray(band).astype('S')
  constellation = np.asarray(constellation).astype('S')
  return reduce(np.char.add, [constellation, '-', sat, '-', band])


def normalize(observation):
  """
  Enforces some standard structure for observation DataFrames.  This
  includes making sure the index is a unique satellite identifier.
  
  This function modifies in place.
  """
  # push the index into the variables in case it's required.
  observation.reset_index(inplace=True)
  # create sids from constellation, satellite number and band.
  sids = create_sid(observation['constellation'],
                    observation['sat'],
                    observation['band'])
  # make sure the sids are unique, then store them
  assert np.unique(sids).size == sids.size
  observation.ix[:, 'sid'] = sids
  # switch to using 'sid' as the index
  observation.set_index('sid', inplace=True)

  if np.any(observation['constellation'].values != 'GPS'):
    warnings.warn("Removing all non-GPS satellites.  Multiconstellation"
                  " isn't quite supported yet.")
    not_gps = observation['constellation'].values != 'GPS'
    # setting constellation to nan then dropping them below
    observation.ix[not_gps, 'constellation'] = np.nan

  # drop any observations where variables in 'subset' are nan.
  subset = ['raw_pseudorange', 'constellation']
  subset = observation.columns.intersection(subset)
  observation.dropna(how='any', subset=subset, inplace=True)

  return observation
