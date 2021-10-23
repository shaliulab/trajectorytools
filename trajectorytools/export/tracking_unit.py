from ethoscope.core.tracking_unit import TrackingUnit as EthoscopeTrackingUnit

class TrackingUnit(EthoscopeTrackingUnit):

    def __init__(self, trajectories, frame_time_table, roi, *args, **kwargs):
        logger.info("Initializing tracking unit")
        trajectory = trajectories[:, roi._idx-1, :]
        super().__init__(
            *args,
            tracking_class=ReadingTracker, roi=roi, trajectory=trajectory, frame_time_table=frame_time_table, stimulator=None,
            **kwargs
        )
        