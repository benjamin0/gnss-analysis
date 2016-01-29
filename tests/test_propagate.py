import pytest
import numpy as np
import pandas as pd

from swiftnav import time as gpstime
from swiftnav import observation, track, signal

from gnss_analysis import (synthetic, locations, propagate,
                           ephemeris, observations, dgnss,
                           time_utils)


def test_delta_tof_for_known_position(ephemerides):
  """
  A first order test which ensures that when a known range is
  propagated to a future time, the resulted propagated range
  matches the exepcted range at that time.
  """
  location_ecef = locations.NOVATEL_ABSOLUTE

  first_toa = ephemerides['time'] + np.timedelta64(100, 's')
  first = synthetic.observations_from_toa(ephemerides, location_ecef,
                                          first_toa)
  # we don't account for satellite error here because tot is in system time
  first = ephemeris.add_satellite_state(first, account_for_sat_error=False)

  # propagate forward in time for i number of seconds and compare to
  # expected observations.
  for i in np.linspace(0, 10, 11):
    second_toa = first_toa + np.timedelta64(int(i), 's')
    expected = synthetic.observations_from_toa(ephemerides, location_ecef,
                                               second_toa)
    to_drop = [x for x in expected.columns if x.startswith('ref_')]
    expected.drop(to_drop, axis=1, inplace=True)
    expected = ephemeris.add_satellite_state(expected,
                                             account_for_sat_error=False)

    actual = propagate.delta_tof_propagate(location_ecef, first,
                                            new_toa=second_toa)

    # make sure the carrier phase is very nearly the same.
    np.testing.assert_almost_equal(expected['carrier_phase'].values,
                                   actual['carrier_phase'].values, 1)
    # We don't expect the time of transmission to be perfect, but do
    # expect it to be within a nanosec of the actual value, so we pop
    # it from the data frame in order to use pandas equality compare utils
    # later.
    d_tot = expected.pop('tot') - actual.pop('tot')
    assert np.all(np.abs(d_tot) <= np.timedelta64(1, 'ns'))

    expected.pop('sat_time')
    actual.pop('sat_time')
    pd.util.testing.assert_frame_equal(expected, actual[expected.columns])


@pytest.mark.skipif(True, reason="still sorting this one out.")
def test_matches_make_propagated_sdiffs(ephemerides):

  def make_nav_meas(obs):
    lock_time = np.nan
    # NOTE: this is using the time of transmission NOT the gpstime
    tot = gpstime.GpsTime(**time_utils.datetime_to_tow(obs['time']))
    sid = signal.GNSSSignal(sat=obs.name, band=0, constellation=0)
    nm = track.NavigationMeasurement(raw_pseudorange=obs.raw_pseudorange,
                                     pseudorange=obs.pseudorange,
                                     carrier_phase=obs.carrier_phase,
                                     raw_doppler=obs.get('doppler', np.nan),
                                     doppler=obs.get('doppler', np.nan),
                                     sat_pos=obs[['sat_x', 'sat_y', 'sat_z']],
                                     sat_vel=obs[['sat_v_x', 'sat_v_y', 'sat_v_z']],
                                     snr=obs.cn0,
                                     lock_time=lock_time,
                                     tot=tot,
                                     sid=sid,
                                     lock_counter=obs.lock)
    return nm

  toa = ephemerides['time'].values[0]
  toa += np.timedelta64(100, 's')
  rover_obs = synthetic.observations_from_toa(ephemerides,
                                             locations.NOVATEL_ABSOLUTE,
                                             toa)

  base_toa = toa - np.timedelta64(1, 's')
  base_obs = synthetic.observations_from_toa(ephemerides,
                                             locations.LEICA_ABSOLUTE,
                                             base_toa)

  rover_nm = [make_nav_meas(x) for _, x in rover_obs.iterrows()]
  base_nm = [make_nav_meas(x) for _, x in base_obs.iterrows()]

  base_pos = base_obs[['ref_x', 'ref_y', 'ref_z']].values[0]
  remote_dists = np.linalg.norm(base_obs[['sat_x', 'sat_y', 'sat_z']] - base_pos,
                                axis=1)

  t = gpstime.GpsTime(**time_utils.datetime_to_tow(toa))
  es = [observations.mk_ephemeris(x)
        for _, x in ephemerides.reset_index().iterrows()]

  # These libswiftnav_sdiffs are very different from the sdiffs computed in python

  # TODO: The following requires a non-master build of libswiftnav
#   sdiffs_t = observation.make_propagated_sdiffs_(rover_nm, base_nm,
#                                                  remote_dists, base_pos,
#                                                  es, t)
#
#   sdiffs = dgnss.make_propagated_single_differences(rover_obs, base_obs,
#                                                     locations.LEICA_ABSOLUTE)
#
#   libswiftnav_sdiffs = pd.DataFrame(x.to_dict() for x in sdiffs_t)
#   libswiftnav_sdiffs['sid'] = [x['sat'] for x in libswiftnav_sdiffs['sid']]
#   libswiftnav_sdiffs.set_index('sid', inplace=True)

