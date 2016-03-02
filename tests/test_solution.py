import copy
import pytest
import itertools

from gnss_analysis import solution

import common


def test_solution_doesnt_mutate(synthetic_stationary_observations, null_filter):
  """
  This tests to make sure that the solution thread doesn't modify the
  observation sets that are passed in.
  """
  obs_sets = [x for _, x in itertools.izip(range(3), synthetic_stationary_observations)]
  obs_sets_copy = copy.deepcopy(obs_sets)
  # we don't need the actual solutions, just need to realize them all and compare originals
  solns = list(solution.solution(obs_sets, filter_results=null_filter))

  for used, orig in zip(obs_sets, obs_sets_copy):
    common.assert_observation_sets_equal(used, orig)


def test_solution(synthetic_stationary_observations, null_filter):
  """
  This tests to make sure that the solution thread doesn't modify the
  observation sets that are passed in.
  """
  obs_sets = [x for _, x in itertools.izip(range(3), synthetic_stationary_observations)]
  # make sure the solution function appropriately stores the output of a filter
  for soln in solution.solution(obs_sets, filter_results=null_filter):
    assert 'filter_results' in soln

  # make sure it works with two filters, and that the previous run didn't mutate anything.
  for soln in solution.solution(obs_sets, one_filter=null_filter, two_filter=null_filter):
    assert 'one_filter' in soln
    assert 'two_filter' in soln
    assert 'filter_results' not in soln

