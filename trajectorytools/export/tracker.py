import math

from ethoscope.core.variables import XPosVariable, YPosVariable, XYDistance, IsInferredVariable
from ethoscope.core.data_point import DataPoint
from ethoscope.trackers.trackers import BaseTracker, NoPositionError

class ReadingTracker(BaseTracker):

    def __init__(self, roi, trajectory, frame_time_table, *args, **kwargs):
        self._trajectory = trajectory
        self._frame_time_table = frame_time_table
        self._old_pos = None

        super().__init__(roi, *args, **kwargs)


    def _find_position(self, t):
        return self._track(t=t)

    def track(self, t, img=None):
        """
        Locate the animal in a image, at a given time.

        :param t: time in ms
        :type t: int
        :param img: the whole frame.
        :type img: :class:`~numpy.ndarray`
        :return: The position of the animal at time ``t``
        :rtype: :class:`~ethoscope.core.data_point.DataPoint`
        """

        self._last_time_point = t
        try:

            points = self._find_position(t)
            if not isinstance(points, list):
                raise Exception("tracking algorithms are expected to return a LIST of DataPoints")

            if len(points) == 0:
                return []

            # point = self.normalise_position(point)
            self._last_non_inferred_time = t

            for p in points:
                p.append(IsInferredVariable(False))

        except NoPositionError:
            if len(self._positions) == 0:
                return []
            else:

                points = self._infer_position(t)

                if len(points) == 0:
                    return []
                for p in points:
                    p.append(IsInferredVariable(True))

        self._positions.append(points)
        self._times.append(t)


        if len(self._times) > 2 and (self._times[-1] - self._times[0]) > self._max_history_length:
            self._positions.popleft()
            self._times.popleft()

        return points

    def _track(self, t):

        frame_idx = self.get_frame_idx(t)
        x, y = self._trajectory._s[frame_idx, :]

        # _, _, w_im, _ = self._roi.rectangle

        pos = x +1.0j * y
        if self._old_pos is None:
            self._old_pos = pos
            raise NoPositionError
        else:
            diff = abs(pos - self._old_pos)

        self._old_pos = pos

        xy_dist = round(math.log10(1. + diff) * 1000)


        x_var = XPosVariable(int(round(x)))
        y_var = YPosVariable(int(round(y)))
        distance = XYDistance(int(xy_dist))

        out = DataPoint([x_var, y_var, distance])

        return [out]


    def get_frame_idx(self, t):
        return self._frame_time_table.loc[self._frame_time_table["frame_time"] == t]["frame_number"].values[0]

