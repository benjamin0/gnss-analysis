import logging
import numpy as np


def can_compute_position(obs):
  """
  Convenience function that returns true if the state contains
  enough information to compute a position.
  """
  # make sure we have enough rover observations
  if obs.empty or obs.shape[0] < 4:
    logging.info("Waiting for 4 satellites, only %d found so far."
                 % obs.shape[0])
    return False
  return True


def get_unique_value(x):
  """
  Convenience function that tries to reduce x to a single
  unique value.
  """
  uniq = np.unique(x)
  assert uniq.size == 1
  return uniq[0]


class PositionEstimator(object):

  def update(self, obs_set):
    """
    Should accept an observation set, perform the computation heavy
    operations and return True or False depending on success.
    """
    raise NotImplementedError("%s needs to implement update()"
                              % self.__class__)

  def get_position(self, obs_set):
    """
    Should accept an observation_set and return the position estimate
    conditional on all data that has been used to update.  This
    should return a DataFrame whose index should be the epoch of
    the obs_set
    """
    raise NotImplementedError("%s needs to implement get_position()"
                              % self.__class__)
