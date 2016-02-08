import pytest
import numpy as np

from gnss_analysis import synthetic, locations, ephemeris, time_utils
from gnss_analysis import constants as c


def test_observation(ephemerides):
  """
  This tests to make sure that the observations returned
  by synthetic.observation are in accordance with RINEX
  defined observation types.
  """
  toa = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')

  obs = synthetic.observation(ephemerides,
                              locations.NOVATEL_ABSOLUTE,
                              toa)
  obs = ephemeris.add_satellite_state(obs, account_for_sat_error=True)

  tof = ephemeris.time_of_flight_from_toa(ephemerides, toa,
                                          obs[['ref_x', 'ref_y', 'ref_z']])
  # the actual time of flight should have been accurately
  # inferred.
  np.testing.assert_allclose(obs['tof'], tof)
  # The time of transmission (after correction) should have been
  # corrected for satellite clock error, here we make sure
  expected_tot = toa - time_utils.timedelta_from_seconds(tof)

  tdiff = time_utils.seconds_from_timedelta(expected_tot - obs['tot'])
  assert np.all(np.abs(tdiff) <= 1e-9)
  # after correction pseudorange should have taken into account
  # satellite clock error, so should match the actual range.
  np.testing.assert_allclose(obs['pseudorange'], tof * c.GPS_C, 6)
  # make sure the raw_pseudorange includes satellite error.
  np.testing.assert_allclose(obs['raw_pseudorange'],
                             (tof - obs['sat_clock_error']) * c.GPS_C)

  # Now we check to see if the doppler is accurate by doing
  # comparing the time differenced carrier phase doppler from
  # two temporally neighboring observations.
  dt = np.timedelta64(1, 'ms')
  toa_next = toa + dt
  obs_next = synthetic.observation(ephemerides,
                                   locations.NOVATEL_ABSOLUTE,
                                   toa_next)
  dt_sec = time_utils.seconds_from_timedelta(dt)
  # We don't expect tdcp_doppler to match exactly, the observed
  # doppler is computed from the satellite velocities and tdcp_doppler
  # is from finite differencing.  For now we can get 0.05 cycle accuracy.
  tdcp_doppler = (obs['carrier_phase'] - obs_next['carrier_phase']) / dt_sec
  np.testing.assert_allclose(obs['doppler'], tdcp_doppler, atol=0.05)


def test_piksi_like_observation(ephemerides):
  toa = np.max(ephemerides['toc'].values) + np.timedelta64(10, 's')

  obs = synthetic.piksi_like_observation(ephemerides,
                                         locations.NOVATEL_ABSOLUTE,
                                         toa)
  obs = ephemeris.add_satellite_state(obs, account_for_sat_error=True)

  rinex_obs = synthetic.observation(ephemerides,
                                    locations.NOVATEL_ABSOLUTE,
                                    toa)
  # make sure that pseudoranges are approximately the same
  np.testing.assert_allclose(rinex_obs['raw_pseudorange'],
                             obs['raw_pseudorange'], 4)
  # make sure that carrier_phases are approximately the same
  np.testing.assert_allclose(rinex_obs['carrier_phase'],
                             obs['carrier_phase'], 2)
