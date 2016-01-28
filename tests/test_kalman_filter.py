
import py.test
import numpy as np

from gnss_analysis.filters import kalman_filter


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
