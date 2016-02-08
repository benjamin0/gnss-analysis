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
