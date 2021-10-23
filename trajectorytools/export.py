import os.path
import tempfile
import json
import datetime
import logging
import traceback
import math
logger = logging.getLogger(__name__)

from joblib import Parallel, delayed
import yaml
import sqlite3
import tqdm
from tqdm.auto import tqdm as tqdm_auto
import numpy as np
import imgstore
import pandas as pd

from trajectorytools import Trajectories
from ethoscope.utils.io import SQLiteResultWriter 
from ethoscope.core.roi import ROI
from ethoscope.web_utils.helpers import get_machine_id
from ethoscope.core.variables import XPosVariable, YPosVariable, XYDistance, IsInferredVariable
from ethoscope.core.data_point import DataPoint
from ethoscope.core.tracking_unit import TrackingUnit as EthoscopeTrackingUnit
from ethoscope.trackers.trackers import BaseTracker, NoPositionError
# from ethoscope.stimulators.stimulators import DefaultStimulator
# from ethoscope.hardware.interfaces.interfaces import HardwareConnection


class ProgressParallel(Parallel):
    #https://stackoverflow.com/a/61027781/3541756

    def __call__(self, *args, **kwargs):
        with tqdm_auto() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self, *args, **kwargs):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
        super().print_progress(*args, **kwargs)


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

        xy_dist = round(math.log10(1. + diff) * 1000)


        x_var = XPosVariable(int(round(x)))
        y_var = YPosVariable(int(round(y)))
        distance = XYDistance(int(xy_dist))

        out = DataPoint([x_var, y_var, distance])

        return [out]


    def get_frame_idx(self, t):
        return self._frame_time_table.loc[self._frame_time_table["frame_time"] == t]["frame_number"].values[0]



class TrackingUnit(EthoscopeTrackingUnit):

    def __init__(self, trajectories, frame_time_table, roi, *args, **kwargs):
        logger.info("Initializing tracking unit")
        trajectory = trajectories[:, roi._idx-1, :]
        super().__init__(
            *args,
            tracking_class=ReadingTracker, roi=roi, trajectory=trajectory, frame_time_table=frame_time_table, stimulator=None,
            **kwargs
        )
        
def get_config(tr: Trajectories):
    """
    Read the idtrackerai config of the experiment whose trajectories are being analyzed
    """
    experiment_folder = os.path.join("/", *tr.params["path"].strip("/").split("/")[::-1][3:][::-1])
    date_time = os.path.basename(experiment_folder)
    config_file = os.path.join(experiment_folder, date_time + ".conf")

    with open(config_file, "r") as fh:
        config = json.load(fh)
    return config

def get_rois(config):
    """
    Return a collection of identical ROIs, one per animal,
    to make the data compatible with ethoscope
    """

    n_individuals = int(config["_nblobs"]["value"])
    ct=np.array(eval(config["_roi"]["value"][0][0]))
    rois = [ROI(ct, i+1) for i in range(n_individuals)]
    return rois


def get_output_filename(output, chunk):
    if chunk is None:
        return output
    else:
        return output.strip(".db") + f"_{str(chunk).zfill(6)}.db"


class ExportMonitor:

    def __init__(self, trajectories, store, output, chunks, *args, frame_range=None, **kwargs):
        
        self._trajectories = trajectories
        self._store = store
        self._output = output
        self._chunks = chunks
        self._frame_time_table = pd.DataFrame(store.get_frame_metadata())


        if chunks[0] != 0:
            n_frames = store._index.get_chunk_metadata(chunks[0]-1)["frame_number"][-1]
            missing_data = np.array([[[[0, ] * trajectories.s.shape[2], ] * trajectories.s.shape[1], ] * n_frames])[0,:,:,:]
            trajectories_w_missing_data = Trajectories.from_positions(missing_data)
            trajectories_w_missing_data.extend(trajectories)
        else:
            trajectories_w_missing_data = trajectories

        
        config = get_config(trajectories)
        rois = get_rois(config)

        self._unit_trackers = [TrackingUnit(trajectories=trajectories_w_missing_data, frame_time_table=self._frame_time_table, roi=r) for r in rois]


    def get_chunk_frame_range(self, chunk):
        return np.array(self._store._index.get_chunk_metadata(chunk)["frame_number"])[[0, -1]].tolist()



    def start(self, ncores=1):

        store_filename = os.path.join(self._store.filename, "metadata.yaml")
        chunks = self._chunks

        if ncores == 1:
            frame_range = self._frame_time_table["frame_number"].iloc[[0,-1]].values.tolist()
            output = [self.start_single_thread(
                trajectories=self._trajectories,
                unit_trackers=self._unit_trackers,
                output=self._output,
                frame_range=frame_range,
                frame_time_table=self._frame_time_table,
                store_filename=store_filename, chunk=None,
                ncores=ncores
            )]
        else:
            frame_ranges = ([None,] * chunks[0]) + [self.get_chunk_frame_range(chunk) for chunk in chunks]
            output = ProgressParallel(n_jobs=ncores, verbose=10)(
                delayed(self.start_single_thread)(
                    trajectories=self._trajectories,
                    unit_trackers=self._unit_trackers,
                    output=self._output,
                    frame_time_table=self._frame_time_table,
                    frame_range=frame_ranges[chunk],
                    store_filename=store_filename, chunk=chunk,
                    ncores=ncores
                ) for chunk in chunks
            )


        

    @staticmethod
    def start_single_thread(trajectories, unit_trackers, output, frame_range, frame_time_table, store_filename, chunk=None, ncores=1):

        trajectories = trajectories[frame_range[0]:frame_range[1], :, :]
        output=get_output_filename(output, chunk)
        thread_safe_store = imgstore.new_for_filename(store_filename)
        
        rw = EthoscopeExport.from_trajectories(
            trajectories,
            thread_safe_store,
            output=output,
            path=output
        )

        if ncores==1:
            iterable = tqdm.tqdm(range(*frame_range))
        else:
            iterable = range(*frame_range)

        with rw as result_writer:

            try:
                for frame_number in iterable:

                    frame_timestamp = frame_time_table.loc[frame_time_table["frame_number"] == frame_number]["frame_time"].values[0]
                    # img, (frame_number, frame_timestamp) = thread_safe_store.get_image(i)

                    t_ms = frame_timestamp

                    for j, track_u in enumerate(unit_trackers):
                        data_rows = track_u.track(t_ms, img=None)
                        if len(data_rows) == 0:
                            continue

                        result_writer.write(t_ms, track_u.roi, data_rows)

                    result_writer.flush(t=t_ms, frame=None, frame_idx=frame_number)
                
                return 0
            
            except Exception as error:
                logger.error(error)
                logger.error(traceback.print_exc())
                return 1





class EthoscopeExport(SQLiteResultWriter):

    def __init__(self, trajectories, store, *args, frame_range=None, **kwargs):

        self._trajectories = trajectories
        super().__init__(*args, **kwargs)


    @classmethod
    def from_trajectories(cls, trajectories, store, output, *args, **kwargs):

        store_file = os.path.join(store.filename, "metadata.yaml")
        with open(store_file, 'r') as f:
            store_metadata = yaml.load(f, Loader=yaml.SafeLoader)

        start_time = store_metadata["__store"]['created_utc']
        start_time = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f").timestamp()

        machine_name = "FLYHOSTEL_1"
        machine_id = get_machine_id()
        version = get_commit_hash()
        db_credentials = {
            "name": output,
            "user": "ethoscope",
            "password": "ethoscope"
        }

        metadata = {
            "machine_id": machine_id,
            "machine_name": machine_name,
            "date_time": start_time,  # the camera start time is the reference 0
            "frame_width": store_metadata["__store"]['imgshape'][1],
            "frame_height": store_metadata["__store"]['imgshape'][0],
            "version": version,
            "experimental_info": "",
            "selected_options": str(store_metadata["__store"]),
        }

        config = get_config(trajectories)
        rois = get_rois(config)

        result_writer = cls(
            trajectories, store,
            db_credentials, rois,
            metadata=metadata,
            take_frame_shots=False, sensor=None,
            make_dam_like_table=False,
            period=10,
            *args, **kwargs
        )

        return result_writer


    def release(self):
        self._queue.put("DONE")



def get_commit_hash():
    hash_str = "v1.0"
    return hash_str


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="ethoscope-like sqlite3 .db file")
    return ap

def main(args=None):

    if args is None:

        ap = get_parser()
        args = ap.parse_args()

    raise NotImplementedError("Need to implement passing of a trajectories file or an experiment-folder")
    trajectories = None # provide a trajectorytools.Trajectories instance
    # path = tempfile.NamedTemporaryFile(suffix=".db").name
    path = args.output

    result_writer = EthoscopeExport.from_trajectories(trajectories, path)
    result_writer.start()
    result_writer.release()

if __name__ == "__main__":
    main()