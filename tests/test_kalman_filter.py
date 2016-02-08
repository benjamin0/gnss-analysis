import pytest
import numpy as np

from gnss_analysis.filters import kalman_filter

@pytest.mark.parametrize('filter_class', [kalman_filter.StaticKalmanFilter,
                                          kalman_filter.DynamicKalmanFilter])
@pytest.mark.parametrize("drop_ref", [True, False])
def test_kalman_drops_sat(synthetic_stationary_states,
                          drop_ref, filter_class):
  """
  Runs through several epochs of data running three filters
  side by side.  One filter receives all the observations,
  another has one will see alternating added/dropped satellites,
  and another will have simultaneously dropped/added satellites.
  
  The baselines are compared, but allowed to vary slightly.
  """
  np.random.seed(1982)
  epochs = [x for _, x in zip(range(10), synthetic_stationary_states)]

  expected_filter = filter_class()
  everyother_filter = filter_class()
  simultaneous_filter = filter_class()

  for i, epoch in enumerate(epochs):
    expected_filter.update(epoch)

    # drop a random satellite and store it in a new frame,
    # but only do so after the first iteration
    if i > 0:
      epoch_dropped = epoch.copy()
      cur_ref = everyother_filter.get_reference_satellite()
      if drop_ref:
        to_drop = cur_ref
      else:
        non_ref = everyother_filter.active_sids.drop(cur_ref)
        to_drop = non_ref.values[np.random.randint(non_ref.size)]
      epoch_dropped['rover'] = epoch_dropped['rover'].drop(to_drop)
    else:
      epoch_dropped = epoch

    # This will drop a satellite one iteration, then not
    # drop it the next (so it will be added back).  Then
    # dro another after.
    if i > 0 and np.mod(i, 0):
      everyother_filter.update(epoch_dropped)
    else:
      everyother_filter.update(epoch)

    # this filter will see a simultaneously added and dropped
    # satellite.
    simultaneous_filter.update(epoch_dropped)


    expected_bl = expected_filter.get_baseline(epoch)
    actual_bl = everyother_filter.get_baseline(epoch)
    simul_bl = simultaneous_filter.get_baseline(epoch)

    # We don't expect a perfect match since each filter will have
    # computed a solution using different data.
    np.testing.assert_allclose(expected_bl, actual_bl, atol=1e-3)
    np.testing.assert_allclose(expected_bl, simul_bl, atol=1e-2)

@pytest.mark.parametrize('filter_class', [kalman_filter.StaticKalmanFilter,
                                          kalman_filter.DynamicKalmanFilter])
def test_kalman_change_reference_sat(synthetic_stationary_states, filter_class):
  """
  This runs through several epochs of data and performs a
  few sanity checks.  In particular it checks that changing
  the reference doesn't change the baseline solution much, and
  that changing the reference and changing right back doesn't
  cause the solution at all.
  """
  np.random.seed(1982)
  epochs = [x for _, x in zip(range(10), synthetic_stationary_states)]

  expected_filter = filter_class()
  actual_filter = filter_class()
  roundtrip_filter = filter_class()

  for epoch in epochs:
    expected_filter.update(epoch)
    actual_filter.update(epoch)
    roundtrip_filter.update(epoch)

    cur_ref = actual_filter.get_reference_satellite()
    non_ref = actual_filter.active_sids.drop(cur_ref)
    new_ref = non_ref.values[np.random.randint(non_ref.size)]
    # change to the new (randomly selected) reference
    actual_filter.change_reference_satellite(new_ref=new_ref)

    # change to the new reference, then change right back.
    roundtrip_filter.change_reference_satellite(new_ref)
    roundtrip_filter.change_reference_satellite(cur_ref)

    # get the baselines
    roundtrip_bl = roundtrip_filter.get_baseline(epoch)
    expected_bl = expected_filter.get_baseline(epoch)
    actual_bl = actual_filter.get_baseline(epoch)

    np.testing.assert_allclose(expected_bl, roundtrip_bl, atol=1e-6)
    np.testing.assert_allclose(expected_bl, actual_bl, atol=1e-6)


def test_kalman_predict_ekf():
  """
  Creates a linear kalman filter the equivalent extended kalman
  filter and ensures that using the two different operator formats
  results in the same output from kalman_predict.
  """
  n = 10
  x = np.random.normal(size=n)
  np.random.seed(1982)
  F = np.random.permutation(np.eye(n))

  def f(z):
    return np.dot(F, z)

  P = np.random.normal(size=(n, n))
  P = np.dot(P, P.T)
  Q = np.random.normal(size=(n, n))
  Q = np.dot(Q, Q.T)

  x_ekf, P_ekf = kalman_filter.kalman_predict(x, P, (f, F), Q)
  x_lkf, P_lkf = kalman_filter.kalman_predict(x, P, F, Q)

  np.testing.assert_array_equal(x_ekf, x_lkf)
  np.testing.assert_array_equal(P_ekf, P_lkf)


def test_kalman_predict():
  # the size of x
  n = 10
  # the size of u
  m = 3

  x = np.zeros(n)
  P = np.eye(n)
  F = np.eye(n)
  Q = np.zeros(n)
  B = np.zeros((n, m))
  u = np.zeros(m)
  # In this situation we expect x and P to remain unchanged
  x_new, P_new = kalman_filter.kalman_predict(x, P, F, Q, B, u)
  np.testing.assert_array_equal(x_new, x)
  np.testing.assert_array_equal(P_new, P)

  # Now we add some process noise, so x should remain unchanged
  # but the covariance P should increase.  Also make sure B and
  # u default to None
  Q = np.eye(n)
  x_new, P_new = kalman_filter.kalman_predict(x, P, F, Q)
  np.testing.assert_array_equal(x_new, x)
  eigs = np.linalg.eigvals(P)
  new_eigs = np.linalg.eigvals(P_new)
  assert np.all(new_eigs > eigs)

  # Now use a non-trivial transformation matrix, F.
  np.random.seed(1982)
  x = np.random.normal(size=n)
  F = np.random.permutation(F)
  np.testing.assert_allclose(np.abs(np.linalg.eigvals(F)), np.ones(n))
  # x_new should be the state transformation applied to x
  x_new, P_new = kalman_filter.kalman_predict(x, P, F, Q, B, u)
  np.testing.assert_array_equal(x_new, np.dot(F, x))
  # P should still increase
  eigs = np.sort(np.linalg.eigvals(P))
  new_eigs = np.sort(np.linalg.eigvals(P_new))
  assert np.all(new_eigs > eigs)

  # Try with some random non-trival covariance matrices
  P = np.random.normal(size=(n, n))
  P = np.dot(P, P.T)
  Q = np.random.normal(size=(n, n))
  Q = np.dot(Q, Q.T)
  x_new, P_new = kalman_filter.kalman_predict(x, P, F, Q, B, u)
  eigs = np.sort(np.linalg.eigvals(P))
  new_eigs = np.sort(np.linalg.eigvals(P_new))
  # the covariance of P should increase (though we permuted so need to sort)
  assert np.all(new_eigs > eigs)
  np.testing.assert_array_equal(x_new, np.dot(F, x))


def test_kalman_update_ekf():
  """
  Creates a linear kalman filter the equivalent extended kalman
  filter and ensures that using the two different operator formats
  results in the same output from kalman_update.
  """
  # the size of x
  n = 10
  # the size of y
  m = 3
  x = np.random.normal(size=n)
  np.random.seed(1982)
  F = np.random.permutation(np.eye(n))

  def f(z):
    return np.dot(F, z)

  P = np.random.normal(size=(n, n))
  P = np.dot(P, P.T)
  R = np.random.normal(size=(m, m))
  R = np.dot(R, R.T)

  H = np.random.normal(size=(m, n))
  y = np.random.normal(size=m)
  def h(z):
    return np.dot(H, z)

  x_ekf, P_ekf = kalman_filter.kalman_update(x, P, y, (h, H), R)
  x_lkf, P_lkf = kalman_filter.kalman_update(x, P, y, H, R)

  np.testing.assert_array_equal(x_ekf, x_lkf)
  np.testing.assert_array_equal(P_ekf, P_lkf)


def test_kalman_update():
  # the size of x
  n = 10
  # the size of y
  m = 3

  x = np.zeros(n)
  P = np.eye(n)
  H = np.random.normal(size=(m, n))
  R = np.eye(m)
  # this will make innovations zero
  y = np.dot(H, x)

  # In this situation we expect x to remain unchanged
  x_new, P_new = kalman_filter.kalman_update(x, P, y, H, R)
  np.testing.assert_array_equal(x_new, x)
  # the posterior distribution should have less noise than
  # the prior.
  eigs = np.linalg.eigvals(P)
  new_eigs = np.linalg.eigvals(P_new)
  # we have to add a fudge factor here due to numerical roundoff
  assert np.all(np.sort(new_eigs) - np.sort(eigs) <= 1e-15)

  # If there is large measurement noise, there should be less
  # of a decrease in the posterior.
  R = 2 * R
  x_new, P_new = kalman_filter.kalman_update(x, P, y, H, R)
  np.testing.assert_array_equal(x_new, x)
  # the posterior distribution should have less noise than
  # the prior.
  larger_eigs = np.linalg.eigvals(P_new)
  # we have to add a fudge factor here due to numerical roundoff
  assert np.all(np.sort(new_eigs) - np.sort(larger_eigs) <= 1e-15)
