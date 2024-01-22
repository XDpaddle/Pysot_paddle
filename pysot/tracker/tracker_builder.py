# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siammask_tracker import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.siamcar_tracker import SiamCARTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker,
          'SiamCARTracker': SiamCARTracker,
         }


def build_tracker(model):
    if cfg.TRACK.TYPE=='SiamCARTracker':
        return TRACKS[cfg.TRACK.TYPE](model, cfg)
    else:
        return TRACKS[cfg.TRACK.TYPE](model)
