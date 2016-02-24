import os
import copy
import pytest
import tempfile
import numpy as np

from gnss_analysis.io import hdf5

import common

def test_to_hdf5_doesnt_modify(datadir, synthetic_stationary_observations):
  """
  Take a couple observation sets, write them to HDF5 and make sure that
  that process doesn't modify anything.
  
  See test_io.py for tests that ensure we can pull the data back out
  properly.
  """
  obs_sets = [y for _, y in zip(range(3),
                                synthetic_stationary_observations)]
  obs_sets_copy = copy.deepcopy(obs_sets)

  tmp_file = os.path.basename(tempfile.mktemp(suffix='.hdf5'))
  output_path = datadir.join(tmp_file).strpath
  hdf5.to_hdf5(obs_sets, output_path)

  for x, y in zip(obs_sets, obs_sets_copy):
    common.assert_observation_sets_equal(x, y)
