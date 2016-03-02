import numpy as np
import pandas as pd



def assert_info_dict_equal(x, y):
  # make sure each dictionary has the same keys
  assert not len(set(x.keys()).symmetric_difference(set(y.keys())))
  # then make sure the values are the same
  for (k, v) in x.iteritems():
    # using np.all allows the values to be lists/arrays and still
    # be properly comparable.
    assert np.all(y[k] == v)


def assert_observation_sets_equal(expected, actual):
  """
  Compares two observation sets each of which should be a dictionary
  holding fields such as 'rover', 'rover_pos', 'rover_info', 'base' ... etc.
  Throws an assertion error if the two differ.
  """
  for k in expected.keys():
    try:
      if k.endswith('_info'):
        info_name = '%s_info' % k
        if info_name in expected:
          assert_info_dict_equal(expected[info_name], actual[info_name])
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
    except AssertionError, e:
      raise AssertionError('In field %s: %s' % (k, e.msg))
