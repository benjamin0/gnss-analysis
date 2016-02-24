import functools

from frozendict import frozendict
from .common import TimeMatchingDGNSSFilter

from gnss_analysis.filters.swiftnav_filter import SwiftNavDGNSSFilter
from gnss_analysis.filters.kalman_filter import DynamicKalmanFilter
from gnss_analysis.filters.kalman_filter import StaticKalmanFilter

# a subsetfunction that removes all but L1 signals from an observation
l1_only = lambda x: x[x.band == '1']

# make this lookup table immutable
lookup = frozendict({'L1-static': functools.partial(StaticKalmanFilter,
                                                    subset_func=l1_only),
                     'static': StaticKalmanFilter,
                     'dynamic': DynamicKalmanFilter,
                     'swiftnav': SwiftNavDGNSSFilter})
