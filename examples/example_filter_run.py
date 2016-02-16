import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from gnss_analysis import filters
from gnss_analysis import solution
from gnss_analysis.io import simulate

if __name__ == "__main__":
  this_dir = os.path.dirname(sys.argv[0])
  test_dir = os.path.join(this_dir, '../tests/test_data')
  rover_file = os.path.join(test_dir, 'cors_drops_reference/seat032/partial_seat0320.16o')
  nav_file = os.path.join(test_dir, 'cors_drops_reference/seat032/seat0320.16n')
  base_file = os.path.join(test_dir, 'cors_drops_reference/ssho032/partial_ssho0320.16o')
  # load a set of observations, each observation set
  # holds any available rover, base and ephemeris data
  # for a given epoch.
  # see also: simulate_from_rinex and simulate_from_hdf5
  obs_sets = simulate.simulate_from_rinex(rover_file,
                                          navigation=nav_file,
                                          base=base_file)
  # define the filter you want to run
  dgnss_filter = filters.StaticKalmanFilter()
  # Then actually run the filter using the observations.
  # The list of solutions consists of a solution for each
  # epoch.  Each solution consists of the observation_set
  # with additional fields corresponding to single point
  # and DGNSS positions.
  solutions = list(solution.solution(obs_sets, dgnss_filter))
  # we can then investigate the results

  def compute_baseline_error(soln):
    # extract the rover position for the current epoch
    rover_pos = np.array([soln['rover_info']['x'],
                          soln['rover_info']['y'],
                          soln['rover_info']['z']])
    # extract the base position for the current epoch
    base_pos = np.array([soln['base_info']['x'],
                         soln['base_info']['y'],
                         soln['base_info']['z']])
    # determine the expected baseline
    expected_baseline = rover_pos - base_pos
    # then the actual baseline from the filter
    actual_baseline = soln['rover_pos'][['baseline_x',
                                         'baseline_y',
                                         'baseline_z']].values[0]
    # and return the difference in the ECEF components
    return actual_baseline - expected_baseline

  # collect all the baseline errors
  errors = [compute_baseline_error(soln) for soln in solutions]
  # and make a plot
  sns.set_style('darkgrid')
  plt.plot(errors)
  plt.xlabel("Epoch")
  plt.ylabel("actual - expected (m)")
  plt.legend(["x", "y", "z"])
  plt.title("Difference between estimated and actual baseline.")
  plt.show()
