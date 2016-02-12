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


def dict_equal(x, y):
  # make sure each dictionary has the same keys
  assert not len(set(x.keys()).symmetric_difference(set(y.keys())))
  # then make sure the values are the same
  for (k, v) in x.iteritems():
    # using np.all allows the values to be lists/arrays and still
    # be properly comparable.
    assert np.all(y[k] == v)


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
    for k in expected.keys():
      if k.endswith('_info'):
        info_name = '%s_info' % k
        if info_name in expected:
          dict_equal(expected[info_name], actual[info_name])
        else:
          assert info_name not in actual
      else:
        if isinstance(expected[k], pd.DataFrame):
          # sometimes the columns are shuffled, which we don't care about,
          # so first we make sure the column sets are the same.  Note
          # that we only care if all the expected values are contained
          # in the actual ones.  If actual contains some extra columns
          # that is considered OK.
          assert not expected[k].columns.difference(actual[k].columns).size
          # then we reorder and compare
          pd.util.testing.assert_frame_equal(expected[k],
                                             actual[k][expected[k].columns])
        else:
          np.testing.assert_array_equal(expected[k], actual[k])
