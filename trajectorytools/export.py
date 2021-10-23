import os.path
import tempfile
import json
import datetime
import logging
import math
logger = logging.getLogger(__name__)

import yaml
import sqlite3
import tqdm
import numpy as np
import imgstore
import pandas as pd

from trajectorytools import Trajectories
from ethoscope.utils.io import SQLiteResultWriter 
from ethoscope.core.roi import ROI
from ethoscope.web_utils.helpers import get_machine_id
from ethoscope.core.variables import XPosVariable, YPosVariable, XYDistance
from ethoscope.core.data_point import DataPoint
from ethoscope.core.tracking_unit import TrackingUnit as EthoscopeTrackingUnit
from ethoscope.trackers.trackers import BaseTracker
# from ethoscope.stimulators.stimulators import DefaultStimulator
# from ethoscope.hardware.interfaces.interfaces import HardwareConnection



class ReadingTracker(BaseTracker):

    def __init__(self, roi, trajectory, store, *args, **kwargs):
        self._trajectory = trajectory
        self._store = store
        self._store_frame_time = pd.DataFrame(store.get_frame_metadata())
        self._old_pos = 0.0+0.0j

        super().__init__(roi, *args, **kwargs)


    def _find_position(self, img, mask, t):
        return self._track(img=img, mask=mask, t=t)

    def _track(self, img, mask, t):

        frame_idx = self._store_frame_time.loc[self._store_frame_time["frame_time"] == t]["frame_number"].values[0]
        x, y = self._trajectory._s[frame_idx, :]

        # _, _, w_im, _ = self._roi.rectangle

        pos = x +1.0j * y
        if self._old_pos is None:
            xy_dist = 1        
        else:
            xy_dist = round(math.log10(1. + abs(pos - self._old_pos)) * 1000) 

        self.old_pos = pos

        x_var = XPosVariable(int(round(x)))
        y_var = YPosVariable(int(round(y)))
        distance = XYDistance(int(xy_dist))

        out = DataPoint([x_var, y_var, distance])

        return [out]

class TrackingUnit(EthoscopeTrackingUnit):

    def __init__(self, trajectories, store, roi, *args, **kwargs):
        logger.info("Initializing tracking unit")
        trajectory = trajectories[:, roi._idx-1, :]
        super().__init__(
            *args,
            tracking_class=ReadingTracker, roi=roi, trajectory=trajectory, store=store, stimulator=None,
            **kwargs
        )
        

class EthoscopeExport(SQLiteResultWriter):

    def __init__(self, trajectories, store, *args, frame_range=None, **kwargs):

        self._trajectories = trajectories
        self._store = store
        config = EthoscopeExport.get_config(trajectories)
        rois = EthoscopeExport.get_rois(config)
        self._unit_trackers = [TrackingUnit(trajectories=trajectories, store=store, roi=r) for r in rois]
        
        if frame_range is None:
            frame_range = (0, self._trajectories.s.shape[0]+1)
        
        self._frame_range = frame_range
        super().__init__(*args, **kwargs)


    def get_image(self, idx):
        return self._store.get_image(idx)


    @classmethod
    def from_trajectories(cls, trajectories, output, *args, **kwargs):

        experiment_folder = os.path.join("/", *trajectories.params["path"].strip("/").split("/")[::-1][3:][::-1])
        config = cls.get_config(trajectories)
        rois = cls.get_rois(config)

        store_file = os.path.join(experiment_folder, "metadata.yaml")

        with open(store_file, 'r') as f:
            store_metadata = yaml.load(f, Loader=yaml.SafeLoader)

        store = imgstore.new_for_filename(store_file)

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


    @classmethod
    def get_config(cls, tr: Trajectories):
        """
        Read the idtrackerai config of the experiment whose trajectories are being analyzed
        """
        experiment_folder = os.path.join("/", *tr.params["path"].strip("/").split("/")[::-1][3:][::-1])
        date_time = os.path.basename(experiment_folder)
        config_file = os.path.join(experiment_folder, date_time + ".conf")

        with open(config_file, "r") as fh:
            config = json.load(fh)
        return config

    @classmethod
    def get_rois(cls, config):
        """
        Return a collection of identical ROIs, one per animal,
        to make the data compatible with ethoscope
        """

        n_individuals = int(config["_nblobs"]["value"])
        ct=np.array(eval(config["_roi"]["value"][0][0]))
        rois = [ROI(ct, i+1) for i in range(n_individuals)]
        return rois

    def start(self):

        frame_range = self._frame_range

        for i in tqdm.tqdm(range(*frame_range)):

            img, (frame_number, frame_timestamp) = self._store.get_image(i)
            t_ms = frame_timestamp

            for j, track_u in enumerate(self._unit_trackers):
                data_rows = track_u.track(t_ms, img) 
                self.write(t_ms, track_u.roi, data_rows)

            self.flush(t=t_ms, frame=img, frame_idx=i)


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