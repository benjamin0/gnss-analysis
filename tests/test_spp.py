import copy
import itertools
import numpy as np

from gnss_analysis import ephemeris, time_utils
from gnss_analysis.io import simulate
from gnss_analysis.filters import spp

import common


def test_spp_consistent(synthetic_observations):
  """
  Performs a PVT solve in python and in libswiftnav and makes sure the
  resulting ECEF positions are the same
  """
  # it shouldn't matter here if we use account_for_sat_error or not.
  obs = ephemeris.add_satellite_state(synthetic_observations)
  spp_pos = spp.single_point_position(obs)
  libswiftnav = spp.libswiftnav_calc_PVT(obs)
  # make sure the two agree to mm resolution.
  np.testing.assert_array_almost_equal(libswiftnav['pos_ecef'],
                                       spp_pos[['x', 'y', 'z']].values[0], 3)
  # make sure the clock offsets agree
  np.testing.assert_array_almost_equal(libswiftnav['clock_offset'],
                                       spp_pos['clock_offset'], 4)
  # make sure times are within a 10us of each other.
  assert spp_pos.index.name == 'time'
  tdiff = time_utils.seconds_from_timedelta(libswiftnav['time'] - spp_pos.index.values)
  assert np.abs(tdiff) < 1e-5


def test_spp_matches_piksi(piksi_log_path):

  states = (x for _, x in zip(range(10), simulate.simulate_from_log(piksi_log_path)))
  tested = False
  for state in states:
    if np.any(np.isnan(state['rover']['raw_doppler'])):
      continue
    # The piksi uses the satellite positions at gps system time of transmission,
    # not at satellite assumed time, so we do not take sat error into account here.
    state['rover'] = ephemeris.add_satellite_state(state['rover'], state['ephemeris'])

    py_pos = spp.single_point_position(state['rover'])
    piksi_pos = state['rover_spp_ecef'][['x', 'y', 'z']].values
    # right now 15cm agreement is all we can get (often less than a cm though).
    # This is possibly due to rounding of the pseudoranges during message
    # transmission and the fact that observations are propagated using doppler,
    # but doppler is not sent in the messages.  Doppler can be inferred using
    # tdcp (the same as on the piksi), but this happens using already
    # propagated carrier phases.  It could also be linked to disagreements
    # between piksi and RINEX conventions.
    assert np.linalg.norm(py_pos[['x', 'y', 'z']].values - piksi_pos) < 0.15
    tested = True

  # make sure we actually tested at least one position.
  assert tested


def test_single_point_position(synthetic_observations):
  """
  Using synthetic data with a known position and rover clock err,
  combined with ephemeris data from a log file, this function
  computes the SPP position and makes sure it is aligned with the
  known position and clock err.
  """
  obs = ephemeris.add_satellite_state(synthetic_observations)
  spp_pos = spp.single_point_position(obs)
  # make sure the resulting estimate is within 1cm of the reference
  ref_position = obs.iloc[0][['ref_x', 'ref_y', 'ref_z']].values
  error = spp_pos[['x', 'y', 'z']].values - ref_position
  # If we just remove all the noise we can get sub mm single point positions!
  assert np.linalg.norm(error) < 1e-4

  # Sub nano-second agreement on the time of arrival
  assert spp_pos.index.name == 'time'
  error = time_utils.seconds_from_timedelta(spp_pos.index.values - obs['ref_t'])
  assert np.all(error <= 1e-9)

  # Here just make sure that the reference rover clock error was recovered
  error = spp_pos['clock_offset'] - obs['ref_rover_clock_error'].values[0]
  assert np.linalg.norm(error) < 1e-9


