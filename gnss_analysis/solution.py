import logging
import pandas as pd

from gnss_analysis import ephemeris
from gnss_analysis.filters import spp
from gnss_analysis.filters.common import PositionEstimator


def solution(obs_sets, **filters):
  """
  Runs one or more estimators against a series of observation sets.
  
  Parameters
  ----------
  obs_sets: iterator of observation sets (see simulate.py)
    An iterator that produces observation sets.  These observation
    sets should at minimum contain 'epoch' and 'rover' fields, but
    may require other fields (depending on filters).
  **filters: kwdargs
    filters should be a set of additional arguments that take the
    form 'filter_name=filter'.  'filter' should be a PositionEstimator
    and filter_name is used to store the output in the observation
    set.  Multiple filters can be specified.
    
    The default is equivalent to:
      solution(obs_sets, rover_spp=spp.SinglePointPosition)

    You can run two filters by using:
      solution(obs_sets, filter1=some_filter, filter2=other_filter)
      
  Returns
  --------
  solution_set : generator
    A generator which produces observation sets with the position
    estimates added as additional fields.    
  """
  # The default estimator is the single point position on the rover.
  if not len(filters):
    filters = {'rover_spp': spp.SinglePointPosition()}

  for estimator_name, estimator in filters.iteritems():
    if not isinstance(estimator, PositionEstimator):
      raise ValueError("argument %s given to solution() is not an estimator"
                       % estimator_name)
    logging.debug("Computing estimates using %s" % estimator_name)

  for obs_set in obs_sets:
    # NOTE: this may add overhead, but ensures that the input sequence
    # doesn't get mutated.
    obs_set = obs_set.copy()
    # Here the satellite clock error is taken into account to convert
    # raw_pseudorange to pseudorange and satellite positions are computed
    # since this is possibly time consuming we do it before passing
    # the observation set into the estimators
    obs_set['rover'] = ephemeris.add_satellite_state(obs_set['rover'],
                                                     obs_set['ephemeris'])
    if 'base' in obs_set:
      obs_set['base'] = ephemeris.add_satellite_state(obs_set['base'],
                                                      obs_set['ephemeris'])

    for estimator_name, estimator in filters.iteritems():
      # NOTE!  We aren't copying here to avoid overhead.  That means
      # we need to make sure all estimators don't mutate observation
      # sets.  Also note that it's up to the estimator to decide
      # if it has enough information to estimate the position.
      updated = estimator.update(obs_set)
      estimate = estimator.get_position(obs_set)
      if estimate is None:
        estimate = pd.DataFrame({}, index=[obs_set['epoch']])
      obs_set[estimator_name] = estimate.copy()

    yield obs_set
