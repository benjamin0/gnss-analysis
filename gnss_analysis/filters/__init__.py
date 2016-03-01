import functools

from frozendict import frozendict

from gnss_analysis.filters.swiftnav_filter import SwiftNavDGNSSFilter
from gnss_analysis.filters.kalman_filter import DynamicKalmanFilter
from gnss_analysis.filters.kalman_filter import StaticKalmanFilter

# make this lookup table immutable
# the lookup table should not include an operator symbols (such as '-')
lookup = frozendict({'L1_static': functools.partial(StaticKalmanFilter,
                                                    single_band=True),
                     'multiband_static': functools.partial(StaticKalmanFilter,
                                                           single_band=False),
                     'L1_dynamic': functools.partial(DynamicKalmanFilter,
                                                     single_band=True),
                     'multiband_dynamic': functools.partial(DynamicKalmanFilter,
                                                     single_band=True),
                     'swiftnav': SwiftNavDGNSSFilter})
