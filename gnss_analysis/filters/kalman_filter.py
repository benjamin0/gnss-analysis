
import logging
import numpy as np

from gnss_analysis import dgnss
from gnss_analysis import constants as c

from gnss_analysis.filters import common


class KalmanFilter(common.TimeMatchingDGNSSFilter):
  """
  This version of an Extended Kalman Fiter is based largely off the paper:
  
    Xiao-wen Chang and Christopher C. Paige and Lan Yin
    Code and Carrier Phase Based Short Baseline GPS Positioning: Computational Aspects
  
  The largest obvious difference between this algorithm and a typical
  float filter is the use of householder transformations in place of double
  differences.
  """

  def __init__(self, *args, **kwdargs):
    self.initialized = False
    super(KalmanFilter, self).__init__(*args, **kwdargs)

  def initialize_filter(self, rover_obs, base_obs):
    self.prev_time = common.get_unique_value(rover_obs['time'])
    self.sids = rover_obs.index.intersection(base_obs.index)
    self.x = np.zeros(3 + self.sids.size - 1)
    # This sets our first guess to be within ~1000 km of the base station
    self.P = 5e5 * np.eye(self.x.size)
    self.initialized = True

  def update_filter(self,):
    pass

  def get_baseline(self, state):
    if not self.initialized:
      return None
    else:
      return self.x[:3]

  def observation_model(self, rover_obs, base_obs):
    """
    Builds variables required for the observation model from a pair of rover
    and base observations.
    
    Returns
    -------
    y : np.ndarray
      A (transformed) array of single difference observations that loosely
      corresponds to double differenced pseudoranges and carrier phases.
    H : np.ndarray
      A 2d array that corresponds to the observation model.  It is used
      to compute the expected values of y given the current estimate of x.
    R : np.ndarray
      A 2d array that represents the covariance of noise in the observation
      model.  In otherwords, it tells us how confident we are in the values
      of y.
    """
    sdiffs = self.get_single_diffs(rover_obs, base_obs, propagate_base=False)

    # This is from equation 1 in Chang, Paige Yin, it is essentially
    # the unit vector pointing from the base receiver to each satellite
    # scaled such that omega_e * baseline = single_difference.
    # As they mention, it could be replaced by one but for high precision
    # applications should be taken into account.
    omega_e = dgnss.omega_dot_unit_vector(self.base_pos, base_obs,
                                          self.x[:3])
    # E_k = omega_e / lambda (see equation 8).
    E = omega_e / c.GPS_L1_LAMBDA

    # m is the number of satellites
    m = sdiffs.shape[0]
    # Here we build a house holder transformation P that allows us to
    # remove shared errors (the same way double differencing does) but
    # without introducing correlation.
    u = -1. / np.sqrt(m) * np.ones(m)
    u[0] += 1
    P = np.eye(m) - 2 * np.outer(u, u) / np.inner(u, u)

    # TODO: confirm these values for the measurement standard deviations!
    sig_cp = 0.1
    sig_pr = 10. / c.GPS_L1_LAMBDA
    sig = sig_cp / sig_pr

    # After applying P to the single differences we will still have m
    # observations, but the first will contain unknown errors corresponding
    # to clock and hardware offsets.  As a result, we drop the first
    # row and take subsets of the transform matrix.
    P_bar = P[1:]
    F = P_bar[:, 1:]

    # We convert the pseudorange to wavelengths and scale by the
    # ratio of carrier phase to pseudorange noises, then apply
    # P_bar to get a new observation vector, y, which loosely corresponds to
    # double differenced observations with error covariance sig^2 * I.
    pr_in_wavelengths = sdiffs['pseudorange'].values / c.GPS_L1_LAMBDA
    y = np.concatenate([np.dot(P_bar, sdiffs['carrier_phase'].values),
                        sig * np.dot(P_bar, pr_in_wavelengths)])

    # Now we compute the linear operator that produces our
    # observations y from the current state estimate x.
    #   y = np.dot(H, x) + v ; v ~ N(0, sig_cp)
    PE = np.dot(P_bar, E)
    # H takes a block form [[PE, F], [sig * PE, 0]]
    # the first column of H is simply the double differencing operators
    # computed above.  The second column corresponds to coefficients
    # for the integer abiguities.  Note that these are only applied
    # to the carrier phase.
    H = np.asarray(np.bmat([[PE, F], [sig * PE, np.zeros_like(F)]]))

    # observation noise
    R = sig_cp * np.eye(H.shape[0])

    return y, H, R


  def updated_matched_obs(self, rover_obs, base_obs):
    if not self.initialized:
      self.initialize_filter(rover_obs, base_obs)

    logging.warn("Ignoring any rising/setting satellites")
    base_obs = base_obs.ix[self.sids]
    rover_obs = rover_obs.ix[self.sids]

    n = self.x.size
    # the observation vector, linear operator and observation noise
    y, H, R = self.observation_model(rover_obs, base_obs)
    # the process model.
    F = np.eye(n)
    # process noise
    Q = np.eye(n)
    # the first three values are the location, we don't expect
    # the receiver will be moving more than 140 mph which over
    # a tenth of a second time step comes out to a process
    # noise with standard deviation of two.
    Q[:3] *= 2.
    # the rest are the ambiguities which should never change.  In
    # fact, in future iterations of this model they should be
    # removed from the filter altogether.
    Q[3:] *= 0.001

    x, P = kalman_predict(self.x, self.P, F, Q)
    x, P = kalman_update(x, P, y, H, R)

    self.x = x
    self.P = P


def expand_operator(z):
  """
  The definition of operators used in Linear Kalman Filters (LKFs)
  and Extended Kalman filters (EKFs) are different, but the math is
  largely the same.  This function takes operators that would be used
  in a LKF (ie, a single matrix) and converts it into a function and
   it's jacobian (which is required for EKFs).
   
  Parameters
  ----------
  z : np.ndarray or (function, np.ndarray)
    An operator which can either be a single matrix (in the case
    of a LKF) or function / jacobian tuple (in the case of an
    EKF).
  
  Returns
  ---------
  f : callable
    A function that applies the operator
  J : np.array
    The Jacobian of the function in the vicinity of the current state.
  """
  if isinstance(z, np.ndarray):
    # if the operator is a matrix we assume it is a linear
    # operator and return a function that simply applies the
    # dot product and the linear operator as Jacobian.
    return lambda x: np.dot(z, x), z
  # Otherwise perhaps the operator is already a function
  # jacobian tuple?
  try:
    f, J = z
  except ValueError:
    raise ValueError("Can't interpret z as a filter operator")
  # TODO: eventually we can pass in theano functions and directly
  # compute the jacobian of an arbitrary function in here.
  return f, J


def kalman_predict(x, P, F, Q, B=None, u=None):
  """
  Takes a pair of state and covariance arrays (x, P) corresponding
  to the state at some iteration of a kalman filter, and advances
  the state to the next time step using the transition F, process
  noise Q and optional forcings B *u.
    
  Parameters
  ----------
  x : np.ndarray
    A 1-d array representing the mean of the state estimate at
    some iteration, k-1, of a kalman filter.
  P : np.ndarray
    A 2-d array representing the covariance of the state estimate at
    some iteration, k-1, of a kalman filter.
  F : operator (see expand_operator)
    The state transition model.
  Q : np.ndarray
    A 2-d array representing the noise introduced in the process
    model during one time step.
  B : np.ndarray
    A 2-d array that projects the describes the impact on forcings
    (u) on the state (x)
  u : np.ndarray
    A 1-d array that holds forcings, aka control variables.
    
  Returns
  ---------
  x : np.ndarray
    A 1-d array representing the mean of the a priori state estimate at
    some iteration, k, of a kalman filter.  This is the best estimate
    of x_k given x_{k-1} but without accounting for observations at step k.
  P : np.ndarray
    A 2-d array representing the covariance of the a priori state estimate at
    some iteration, k, of a kalman filter. This is the best estimate
    of P_k given x_{k-1} but without accounting for observations at step k.
  """
  # F should be convertable to an operator pair (function, jacobian) that
  # applies the transition process model, ie x_k = f(x_{k-1}).
  f, F = expand_operator(F)
  # apply the state transition function f
  x = f(x)
  # optionally apply and forcings
  if B is not None and u is not None:
    x += np.dot(B, u)
  # update the covariance of x, this comes in two parts
  # F P F^T, which is the covariance of x_k due to the
  # transition function, and Q which is the noise
  # of the process model.
  P = np.dot(np.dot(F, P), F.T) + Q
  return x, P


def kalman_update(x, P, y, H, R):
  """
  Takes a pair of state and covariance arrays (x, P) corresponding
  to the state at some iteration of a kalman filter, and computes
  and update to the state conditional on a new set of observations
  y.
  
  Parameters
  ----------
  x : np.ndarray
    A 1-d array representing the mean of the state estimate at
    some iteration, k-1, of a kalman filter.
  P : np.ndarray
    A 2-d array representing the covariance of the state estimate at
    some iteration, k-1, of a kalman filter.
  y : np.ndarray
    A 1-d array representing a set of new observations.
  H : operator (see expand_operator)
    The observation model. x' = H(y) + R
  R : np.ndarray
    A 2-d array representing the noise introduced in the measurement process
    
  Returns
  ---------
  x : np.ndarray
    A 1-d array representing the mean of the posterior state estimate at
    some iteration, k, of a kalman filter.  This is the best estimate
    of x_k given x_{k-1} and all observations up to step k.
  P : np.ndarray
    A 2-d array representing the covariance of the posterior state estimate at
    some iteration, k, of a kalman filter. This is the best estimate
    of P_k given x_{k-1} and all observations up to step k.
  """
  h, H = expand_operator(H)
  # actual observation minus the observation model
  innov = y - h(x)
  S = np.dot(np.dot(H, P), H.T) + R
  # NOTE: np.linalg.inv(S) is a horrible HORRIBLE thing to do, but
  #   for the first time around it should be fine.
  K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
  x = x + np.dot(K, innov)
  P = np.dot(np.eye(P.shape[0]) - np.dot(K, H), P)
  return x, P
