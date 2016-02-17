import os
import copy
import pytest
import tempfile
import itertools
import numpy as np
import pandas as pd

from gnss_analysis import ephemeris
from gnss_analysis.io import hdf5
from gnss_analysis.io import simulate

import common


@pytest.fixture(params=['raw', 'add_state', 'solution'])
def observations_and_solutions(observation_sets,
                               request):

  if request.param == 'raw':
    return observation_sets
  elif request.param == 'add_state':
    # Add satellite state to the observations and return a new generator
    def add_state(obs_set):
      obs_set['rover'] = ephemeris.add_satellite_state(obs_set['rover'],
                                                       obs_set['ephemeris'])
      if 'base' in obs_set:
        obs_set['base'] = ephemeris.add_satellite_state(obs_set['base'],
                                                         obs_set['ephemeris'])
      return obs_set

    return (add_state(obs_set) for obs_set in observation_sets)
  elif request.param == "solution":
    from gnss_analysis import solution
    from gnss_analysis.filters import StaticKalmanFilter
    return solution.solution(observation_sets, StaticKalmanFilter())
  else:
    raise NotImplementedError("Unknown param %s" % request.param)


@pytest.mark.slow
def test_roundtrip_to_hdf5(observations_and_solutions,
                           datadir):
  """
  Reads from a piksi log, dumps to HDF5 then loads
  back and makes sure the two observation set iterators
  are identical.
  """
  tmp_file = os.path.basename(tempfile.mktemp(suffix='.hdf5'))
  output_path = datadir.join(tmp_file).strpath
  # only use the first 10 observation sets
  obs_sets = [x for _, x in zip(range(10), observations_and_solutions)]
  # make a copy of them so we aren't accidentally modifying them
  # before comparison
  obs_sets_copy = copy.deepcopy(obs_sets)
  # save the copy to hdf5
  hdf5.to_hdf5(obs_sets_copy, output_path)
  # and then pull them back out of hdf5
  hdf5_sets = [x for x in simulate.simulate_from_hdf5(output_path)]
  # and make sure we haven't changed anything

  for expected, actual in itertools.izip_longest(obs_sets, hdf5_sets):
    common.assert_observation_sets_equal(expected, actual)
