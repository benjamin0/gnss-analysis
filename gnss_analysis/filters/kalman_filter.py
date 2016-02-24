import scipy
import logging
import numpy as np
import pandas as pd

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

  def __init__(self, subset_func=None, *args, **kwdargs):
    self.initialized = False
    if subset_func is None:
      subset_func = lambda x: x
    self.subset_func = subset_func
    super(KalmanFilter, self).__init__(*args, **kwdargs)

  def initialize_filter(self, rover_obs, base_obs):
    """
    Need to implement this in a subclassed filter.

    """
    return NotImplementedError()

  def process_model(self):
    """
    Need to implement this in a subclassed filter.

    """
    raise NotImplementedError()

  def get_baseline(self, obs_set):
    """
    Returns the baseline corresponding to the current state.
    """
    # TODO: In theory we could return the baseline for future / past epochs
    if not self.initialized:
      return None
    else:
      # For now we make just return the current filter state.
      return self.x.iloc[:self.n_dim]

  def choose_reference_sat(self, new_sids):
    """
    Uses the current set of active sids and the set of new sids to
    choose a new reference satellite (one that is active in both
    sets).
    """
    # For now we just set the new reference to be the next
    # common satellite.  Note at this point the active sids
    # actually corresponds to the active sids at the last iteration.
    common_sids = self.active_sids.intersection(new_sids)
    # Just grab the first one in common
    # TODO: maybe choose based off of signal strength?
    return common_sids.values[0]


  def get_reference_satellite(self):
    """
    Returns the sid for the reference satellite.
    """
    # The current reference is the satellite that isn't included
    # in the integer ambiguity index.
    cur_ref = self.active_sids.difference(self.x.index[self.ambiguity_states_idx:])
    # There should only ever be one such satellite.
    return common.get_unique_value(cur_ref)


  def change_reference_satellite(self, new_ref=None, new_sids=None):
    """
    Performs a transformation that swaps the old reference satellite
    for a new reference.  The filter state and covariance are
    then modified in place.
    """
    # infer the current reference satellite
    cur_ref = self.get_reference_satellite()

    # choose the new reference if it wasn't specified
    if new_ref is None:
      if new_sids is None:
        raise ValueError("Need either a new reference or new sat ids to"
                         " decide which satellite to use as reference.")
      new_ref = self.choose_reference_sat(new_sids)

    # make sure the new reference was around last time.
    assert new_ref in self.active_sids.values

    # avoid unnecessary computation
    if new_ref == cur_ref:
      logging.warn("Tried changing reference satellite to the same"
                   " satellite.  This implies faulty logic elsewhere")
      return

    logging.debug("Changing reference from %s to %s" % (cur_ref, new_ref))

    # determine which index in x corresponds to the new reference
    new_ind = self.x.index.get_indexer([new_ref])

    # build an orthogonal matrix that will swap references, this is
    # done by subtracting out the ambiguities corresponding to the
    # new reference from the rest of the ambiguities:
    #
    # N_1 - N_0            N_1 - N_j  = (N_1 - N_0) - (N_j - N_0)
    #    ..          K
    # N_j - N_0    --->    N_0 - N_j  = -(N_j - N_0)
    #    ..
    # N_n - N_0            N_n - N_j  = (N_n - N_0) - (N_j - N_0)
    #
    # Which is accomplished by creating an identity matrix, then
    # replacing the column corresponding to the new reference with
    # negative ones.
    K = np.eye(self.x.size)
    # The first three indices are the position, we don't want
    # to mess with those.
    K[self.ambiguity_states_idx:, new_ind] = -1.
    self.K = K

    # Apply the transformation to swap the cur_ref for the new one
    self.x.values[:] = np.dot(K, self.x)
    # reflect the change in the index of x
    new_index = self.x.index.insert(new_ind, cur_ref)
    new_index = new_index.delete(new_ind + 1)
    self.x.index = new_index

    # And apply the transformation to the covariance matrix as well.
    self.P = np.dot(np.dot(K, self.P), K.T)


  def drop_satellites(self, to_drop, new_sids):
    """
    Drops satellites from the filter state/covariance, if any
    of the satellites being dropped are reference satellites
    the reference is first changed to another active one, then dropped.
    """
    # If the dropped satellite is not in the state (x) then
    # it must be the reference satellite.  In which case we
    # must first change the reference.
    if not to_drop.isin(self.x.index).any():
      self.change_reference_satellite(new_sids=new_sids)

    # Iteratively drop satellites from the state and rows/cols from
    # the covariance matrix.
    for drop in to_drop:
      ind = common.get_unique_value(self.x.index.get_indexer([drop]))
      self.x = self.x.drop(drop)
      self.P = np.delete(np.delete(self.P, ind, axis=0), ind, axis=1)
      self.active_sids = self.active_sids.drop(drop)


  def add_satellites(self, to_add):
    """
    Updates the filter state and covariance to contain new satellites.
    """
    self.x = self.x.append(pd.Series([0], index=to_add))
    new_P = np.zeros(np.array(self.P.shape) + to_add.size)
    new_P[:-to_add.size, :-to_add.size] = self.P
    new_P[-to_add.size:, -to_add.size:] = (self.sig_init_p / c.GPS_L1_LAMBDA) * np.eye(to_add.size)
    self.P = new_P
    self.active_sids = self.active_sids.append(to_add)


  def update_active_satellites(self, new_sids):
    """
    This takes care of add/drop satellite logic.
    """
    # the full set of tracked sids is the set of active ones
    # (which includes the reference) and any others tracked
    # in the state (x). For now x should only contain active
    # satellites, but in the future we may want to leave
    # deactivated satellites around.
    all_sids = self.active_sids.union(self.x.index[self.ambiguity_states_idx:])
    # are all the satellites already active
    if np.all(new_sids == self.active_sids):
      return

    # Any satellites that aren't in the set of new satellites
    # but were active must have been lost, so we drop them
    to_drop = self.active_sids.difference(new_sids)
    if to_drop.size:
      self.drop_satellites(to_drop, new_sids)

    # Any satellites that are in the new set but not in the
    # active set must be new so we add them.
    to_add = new_sids.difference(all_sids)
    if to_add.size:
      self.add_satellites(to_add)


  def double_difference_observation_model(self, sdiffs):
    """
    Builds the double differenced (or in this case, orthogonal equivalent
    of the double difference) observation model.

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
    # This is from equation 1 in Chang, Paige Yin, it is essentially
    # the unit vector pointing from the base receiver to each satellite
    # scaled such that omega_e * baseline = single_difference.
    # As they mention, it could be replaced by one but for high precision
    # applications should be taken into account.
    omega_e = dgnss.omega_dot_unit_vector(self.base_pos, sdiffs,
                                          self.x.values[:self.n_dim])
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

    # sig is the ratio of the carrier phase and pseudorange
    # standard deviations in units of cycles.
    sig_ratio = self.sig_cp / (self.sig_pr / c.GPS_L1_LAMBDA)

    # After applying P to the single differences we will still have m
    # observations, but the first will contain unknown errors corresponding
    # to clock and hardware offsets.  As a result, we drop the first
    # row and take a subset of the transform matrix.
    P_bar = P[1:]

    # We convert the pseudorange to wavelengths and scale by the
    # ratio of carrier phase to pseudorange noises, then apply
    # P_bar to get a new observation vector, y, which loosely corresponds to
    # double differenced observations with error covariance sig^2 * I.
    pr_in_wavelengths = sdiffs['pseudorange'].values / c.GPS_L1_LAMBDA
    y = np.concatenate([np.dot(P_bar, sdiffs['carrier_phase'].values),
                        sig_ratio * np.dot(P_bar, pr_in_wavelengths)])

    if np.any(np.isnan(y)):
      import webbrowser
      webbrowser.open('http://www.quickmeme.com/img/c7/c727a64980f4d337c8'
                      '46903ce407fe2ffccfaeecb9c252a3045f6b7a8a10b2a6.jpg')
      raise ValueError("I've made a huge mistake")

    # Now we compute the linear operator that produces our
    # observations y from the current state estimate x.
    #   y = np.dot(H, x) + v ; v ~ N(0, sig_cp)
    PE = np.dot(P_bar, E)
    # As defined by Chang et al F is:
    #   F = I - e e^T / (m - sqrt(m))
    # Note that the size of e in equations 12 and 16 are different.
    F = np.eye(m - 1) - np.ones((m - 1, m - 1)) / (m - np.sqrt(m))
    # H takes a block form [[PE, F], [sig * PE, 0]]
    # the first column of H is simply the double differencing operators
    # computed above.  The second column corresponds to coefficients
    # for the integer abiguities.  Note that these are only applied
    # to the carrier phase.
    n_rows, n_cols = PE.shape
    H = np.asarray(np.bmat([[PE,
                             np.zeros((n_rows, (self.ambiguity_states_idx - self.n_dim))),
                             F],
                            [sig_ratio * PE,
                             np.zeros((n_rows, (self.ambiguity_states_idx - self.n_dim))),
                             np.zeros_like(F)]]))

    # R is the observation noise.  Notice that because we rescaled above
    # our observation noise has a constant diagonal.  Also note that
    # because we've defined sig_cp to be the noise in an observation
    # of carrier_phase and because R is the noise in the double differenced
    # carrier phase, we have to multiply by 4.
    R = 4 * self.sig_cp * np.eye(H.shape[0])
    return y, H, R

  def update_matched_obs(self, rover_obs, base_obs):
    """
    Updates the model given a set of time matched rover and base
    observations.  The update is done in place.
    """
    rover_obs = self.subset_func(rover_obs)
    base_obs = self.subset_func(base_obs)
    if not self.initialized:
      self.initialize_filter(rover_obs, base_obs)

    # keep track of the current time of the filter
    self.cur_time = common.get_unique_value(rover_obs['time'])

    # Compute the single differences.  get_single_diffs is
    # expected to take care of logic such as determining if there
    # was a loss of lock, and dropping such satellites.
    sdiffs = self.get_single_diffs(rover_obs, base_obs, propagate_base=False)

    # Here we check if we've added or lost any satellites
    if sdiffs.index.sym_diff(self.active_sids).size:
      self.update_active_satellites(sdiffs.index)

    # We expect the sdiffs to be in the same order as the ambiguity states
    # Here we simply make sure that's the case
    expected_order = self.x.index[self.ambiguity_states_idx:].insert(0, self.get_reference_satellite())

    if not np.all(sdiffs.index == expected_order):
      sdiffs = sdiffs.ix[expected_order]

    # if these fail we clearly didn't update the active satellites
    # properly.
    assert sdiffs.index.size == self.active_sids.size
    assert np.all(sdiffs.index.isin(self.active_sids))

    # the observation vector, linear operator and observation noise
    y, H, R = self.double_difference_observation_model(sdiffs)
    # the process model
    F, Q = self.process_model()

    # use the process model to predict x_{k|k-1}
    x, P = kalman_predict(self.x.values, self.P, F, Q)
    # use the observation model to form the posterior x_{k|k}
    x, P = kalman_update(x, P, y, H, R)

    # x is a pandas Series, we want to keep the index but fill the values
    self.x.values[:] = x
    # P is an ndarray so we just overwrite it.
    self.P = P


class StaticKalmanFilter(KalmanFilter):

  def __init__(self, sig_x=2., sig_z=0.01, sig_cp=0.02,
               sig_pr=3., sig_init_p=5e5, *args, **kwdargs):
    """
    Initializes a Kalman Filter with a static process model with the option to
    specify observation and process noise.

    Parameters
    ----------
    sig_x : float
      The process noise corresponding to the position estimate (meters)
        x_k|{k-1} = x_{k-1}|{k-1} + N(0, sig_x^2)
    sig_z : float
      The process noise corresponding to the ambiguity estimate. (cycles)
        z_k|{k-1} = z_{k-1}|{k-1} + N(0, sig_z^2)
    sig_cp : float
      The observation noise corresponding to carrier phase. (cycles)
    sig_pr : float
      The observation noise correspondong to pseudorange. (meters)
    sig_init_p : float
      The uncertainity associated with the initial baseline estimate. (meters)

    """
    self.sig_x = sig_x
    self.sig_z = sig_z
    self.sig_cp = sig_cp
    self.sig_pr = sig_pr
    self.sig_init_p = sig_init_p
    self.initialized = False
    # The first 3 states of the filter are our baseline estimates in x, y, & z.
    self.n_dim = 3
    # The ambiguity states start at index 3.
    self.ambiguity_states_idx = 3
    super(StaticKalmanFilter, self).__init__(*args, **kwdargs)

  def initialize_filter(self, rover_obs, base_obs):
    """
    This is inteded to be run once and only once.  It takes an
    initial set of observations and creates the corresponding
    state and covariance matrices
    """
    assert not self.initialized
    self.active_sids = rover_obs.index.intersection(base_obs.index)
    # A series containing the n, e and d components of the baseline (in meters)
    pos = pd.Series(np.zeros(self.n_dim), index=['x', 'y', 'z'])
    # sets the reference satellite to be the first in the active set and
    # creates the state vector of double differenced ambiguities.
    amb = pd.Series(np.zeros(self.active_sids.size - 1),
                    index=self.active_sids[1:])
    # the state vector is a concatenation of the position and ambiguities
    self.x = pd.concat([pos, amb])
    self.P = np.eye(self.x.size)
    # initialize the position covariance
    self.P[:self.n_dim] *= self.sig_init_p
    # and the ambiguity covariance.
    self.P[self.ambiguity_states_idx:] *= self.sig_init_p / c.GPS_L1_LAMBDA
    self.initialized = True

  def process_model(self):
    """
    Returns the process model which consists of the matrix F and Q
    such that:

        x_{k|k-1} = F x_{k-1|k-1} + N(0, Q)
    """
    # It's possible that the observation_model will drop satellites
    # when it detects slips, so this definition of n needs to be after
    n = self.x.size
    # the process model.
    F = np.eye(n)
    # process noise
    Q = np.eye(n)
    # the first three values are the location, we don't expect
    # the receiver will be moving more than 140 mph which over
    # a tenth of a second time step comes out to a process
    # noise with standard deviation of two.
    Q[:self.n_dim] *= self.sig_x**2
    # the rest are the ambiguities which should slowly change
    # TODO: what do we do about cycle slips.
    Q[self.ambiguity_states_idx:] *= self.sig_z**2
    return F, Q


class DynamicKalmanFilter(KalmanFilter):
  """
  A Kalman Filter with a dynamic model. See section 7.6 (page 435) of GPS
  Satellite Surveying 4e by Leick.

  """

  def __init__(self,
               sig_dynamics=1.,
               sig_z=0.01,
               sig_cp=0.02,
               sig_pr=3.,
               sig_init_p=5e5,
               sig_init_v=10.,
               sig_init_a=0.1,
               correlation_time=10.,
               *args, **kwdargs):
    """
    Initializes a Kalman Filter with a dynamic process model.

    Parameters
    ----------
    sig_dynamics : float
      The process noise that is associated with the expected dynamics.
    sig_z : float
      The process noise corresponding to the ambiguity estimate. (cycles)
        z_k|{k-1} = z_{k-1}|{k-1} + N(0, sig_z^2)
    sig_cp : float
      The observation noise corresponding to carrier phase. (cycles)
    sig_pr : float
      The observation noise correspondong to pseudorange. (meters)
    sig_init_p : float
      The uncertainity associated with the initial baseline position estimate. (meters)
    sig_init_v : float
      The uncertainity associated with the initial baseline velocity estimate. (m/s)
    sig_init_a : float
      The uncertainity associated with the initial baseline accleration estimate. (m/s^2)
    correlation_time : float
      A parameter that governs the volatility of acceleration. Greater values
      indicates steadier movement. (seconds)

    """
    self.sig_dynamics = sig_dynamics
    self.sig_z = sig_z
    self.sig_cp = sig_cp
    self.sig_pr = sig_pr
    self.sig_init_p = sig_init_p
    self.sig_init_v = sig_init_v
    self.sig_init_a = sig_init_a
    # Time between measurements (will need to determine this on the fly later)
    self.dt = 1.0
    self.gamma = np.exp(-self.dt / correlation_time)# See eqn 7.6.2 in Leick
    self.initialized = False
    # Number of dimensions we want to find a solution (x, y, z)
    self.n_dim = 3
    # Number of dynamic states (position, velocity, accleration)
    self.dynamic_states = 3
    # Index at which the ambiguity states start
    self.ambiguity_states_idx = 9
    super(DynamicKalmanFilter, self).__init__(*args, **kwdargs)

  def initialize_filter(self, rover_obs, base_obs):
    """
    This is inteded to be run once and only once.  It takes an
    initial set of observations and creates the corresponding
    state and covariance matrices
    """
    assert not self.initialized
    self.active_sids = rover_obs.index.intersection(base_obs.index)
    # A series containing the n, e and d components of the baseline (in meters)
    pos = pd.Series(np.zeros(self.dynamic_states * self.n_dim),
                    index=['x', 'y', 'z', 'x_vel', 'y_vel', 'z_vel', 'x_acc', 'y_acc', 'z_acc'])
    # sets the reference satellite to be the first in the active set and
    # creates the state vector of double differenced ambiguities.
    amb = pd.Series(np.zeros(self.active_sids.size - 1),
                    index=self.active_sids[1:])
    # the state vector is a concatenation of the position and ambiguities
    self.x = pd.concat([pos, amb])
    self.P = np.eye(self.x.size)
    # initialize the position covariance
    self.P[:self.n_dim] *= self.sig_init_p
    self.P[self.n_dim:(2 * self.n_dim)] *= self.sig_init_v
    self.P[(2 * self.n_dim):(3 * self.n_dim)] *= self.sig_init_a
    # and the ambiguity covariance.
    self.P[self.ambiguity_states_idx:] *= self.sig_init_p / c.GPS_L1_LAMBDA

    self.initialized = True

  def process_model(self):
    """
    Returns the process model which consists of the matrix F and Q
    such that:

        x_{k|k-1} = F x_{k-1|k-1} + N(0, Q)

    The process model is described by the following equations where
    x := position, v := velocity, a := acceleration:
    x_{k|k-1} = x_{k-1|k-1} + dt * v_{k-1|k-1} + .5 * dt^2 * a_{k-1|k-1}
    v_{k|k-1} = v_{k-1|k-1} + dt * a_{k-1|k-1}
    a_{k|k-1} = gamma * a_{k-1|k-1}

    In three dimensions, the state transition model for the dynamic states is:
    F_dyanmics = [[eye(3), dt * eye(3), .5 * dt^2 * eye(3)],
                  [0,           eye(3),        dt * eye(3)],
                  [0,                0,     gamma * eye(3)]]

    The state transition model for the ambiguity states is the identity matrix.

    The covariance of the process noise for the dynamic states, Q_dynamics:
    Q_dynamics = [[(dt^5)/20 * eye(3),  (dt^4)/8 * eye(3), (dt^3)/6 * eye(3)],
                  [ (dt^4)/8 * eye(3),  (dt^3)/3 * eye(3), (dt^2)/2 * eye(3)],
                  [ (dt^3)/6 * eye(3),  (dt^2)/2 * eye(3),       dt * eye(3)]]

    The covariance of the process noise for the ambiguity states is diagonal
    with magnitude of sigma_z^2.

    """
    # It's possible that the observation_model will drop satellites
    # when it detects slips, so this definition of n needs to be after
    n = self.x.size
    # the process model.
    F = np.eye(n)
    F_dynamics = np.array(np.bmat([[np.eye(self.n_dim), self.dt * np.eye(self.n_dim), 0.5 * self.dt**2 * np.eye(self.n_dim)],
                                   [np.zeros((self.n_dim, self.n_dim)), np.eye(self.n_dim), self.dt * np.eye(self.n_dim)],
                                   [np.zeros((self.n_dim, self.n_dim)), np.zeros((self.n_dim, self.n_dim)), self.gamma * np.eye(self.n_dim)]]))

    F[:self.ambiguity_states_idx, :self.ambiguity_states_idx] = F_dynamics
    # process noise
    Q = np.eye(n) * self.sig_z**2
    Q_dynamics = self.sig_dynamics**2 \
                 * np.array(np.bmat([[(self.dt**5 / 20) * np.eye(self.n_dim), (self.dt**4 / 8) * np.eye(self.n_dim), (self.dt**3 / 6) * np.eye(self.n_dim)],
                                     [(self.dt**4 / 8) * np.eye(self.n_dim), (self.dt**3 / 3) * np.eye(self.n_dim), (self.dt**2 / 2) * np.eye(self.n_dim)],
                                     [(self.dt**3 / 6) * np.eye(self.n_dim), (self.dt**2 / 2) * np.eye(self.n_dim), self.dt * np.eye(self.n_dim)]]))

    Q[:self.ambiguity_states_idx, :self.ambiguity_states_idx] = Q_dynamics
    return F, Q


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
  PHT = np.dot(P, H.T)
  S = np.dot(np.dot(H, P), H.T) + R
  x = x + PHT.dot(np.linalg.solve(S, innov))
  P = np.dot(np.eye(P.shape[0]) - PHT.dot(np.linalg.solve(S, H)), P)
  return x, P
