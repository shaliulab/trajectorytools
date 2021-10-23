import os.path
import datetime
import yaml

from ethoscope.utils.io import SQLiteResultWriter 
from ethoscope.web_utils.helpers import get_machine_id
from sleep_models.export.helpers import get_commit_hash

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

        machine_name = "FLYHOSTEL_001"
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