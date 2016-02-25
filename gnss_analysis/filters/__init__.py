import functools

from frozendict import frozendict
from .common import TimeMatchingDGNSSFilter

from gnss_analysis.filters.swiftnav_filter import SwiftNavDGNSSFilter
from gnss_analysis.filters.kalman_filter import DynamicKalmanFilter
from gnss_analysis.filters.kalman_filter import StaticKalmanFilter

# make this lookup table immutable
lookup = frozendict({'L1-static': functools.partial(StaticKalmanFilter,
                                                    single_band=True),
                     'multiband-static': functools.partial(StaticKalmanFilter,
                                                           single_band=False),
                     'static': StaticKalmanFilter,
                     'dynamic': DynamicKalmanFilter,
                     'swiftnav': SwiftNavDGNSSFilter})
