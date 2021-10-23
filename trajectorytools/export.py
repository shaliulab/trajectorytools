import os.path
import tempfile
import json
import datetime

import yaml
import sqlite3
import tqdm
import numpy as np
import imgstore

from trajectorytools import Trajectories
from ethoscope.utils.io import SQLiteResultWriter 
from ethoscope.core.roi import ROI
from ethoscope.web_utils.helpers import get_machine_id
from ethoscope.core.variables import XPosVariable, YPosVariable
from ethoscope.core.data_point import DataPoint
# from ethoscope.core.tracking_unit import tracking_unit

class TrackingUnit:

    def __init__(self, trajectories, roi):
        self._trajectory = trajectories[:, roi._idx-1, :]
        self._roi = roi

    @property
    def roi(self):
        return self._roi


    def track(self, frame_idx, absolute=True):

        x_pos, y_pos = self._trajectory._s[frame_idx, :]

        if absolute:
            x,y,w,h = self._roi._rectangle
            x_pos -= x
            y_pos -= y

        x_var = XPosVariable(int(round(x_pos)))
        y_var = YPosVariable(int(round(y_pos)))
        out = DataPoint([x_var, y_var])

        return [out]


class EthoscopeExport(SQLiteResultWriter):

    def __init__(self, trajectories, store, *args, **kwargs):

        self._trajectories = trajectories
        self._store = store
        config = EthoscopeExport.get_config(trajectories)
        rois = EthoscopeExport.get_rois(config)        
        self._unit_trackers = [TrackingUnit(trajectories, r) for r in rois]
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

        for i in tqdm.tqdm(range(self._trajectories.s.shape[0])):

            img, (frame_number, frame_timestamp) = self._store.get_image(i)
            t_ms = frame_timestamp

            for j, track_u in enumerate(self._unit_trackers):
                data_rows = track_u.track(i) 
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