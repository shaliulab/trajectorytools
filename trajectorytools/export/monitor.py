import os.path
import logging
import threading
import traceback

import numpy as np
import imgstore
import pandas as pd
import tqdm

from trajectorytools import Trajectories

from trajectorytools.export.io import EthoscopeExport as ResultWriterClass
from trajectorytools.export.parallel import ProgressParallel, delayed
from trajectorytools.export.helpers import get_config, get_rois, get_output_filename, get_commit_hash
from trajectorytools.export.tracking_unit import TrackingUnit

logger = logging.getLogger(__name__)

class ExportMonitor(threading.Thread):
    """
    Export a trajectorytools.Trajectory object to an Ethoscope-like sqlite3 file
    Parallel processing is available, so one chunk is saved per CPU
    """

    def __init__(self, trajectories, store, output, chunks, *args, frame_range=None, **kwargs):

        self._store = store
        self._output = output
        self._chunks = chunks
        self._frame_time_table = pd.DataFrame(store.get_frame_metadata())
        self._frame_range = frame_range


        if chunks[0] != 0:
            n_frames = store._index.get_chunk_metadata(chunks[0]-1)["frame_number"][-1]
            self._first_frame = store._index.get_chunk_metadata(chunks[0])["frame_number"][0]
            missing_data = np.array([[[[0, ] * trajectories.s.shape[2], ] * trajectories.s.shape[1], ] * n_frames])[0,:,:,:]
            trajectories_w_missing_data = Trajectories.from_positions(missing_data)
            trajectories_w_missing_data.extend(trajectories)
        else:
            trajectories_w_missing_data = trajectories


        self._trajectories = trajectories_w_missing_data
        config = get_config(self.trajectories)
        rois = get_rois(config)

        self._unit_trackers = [TrackingUnit(trajectories=trajectories_w_missing_data, frame_time_table=self._frame_time_table, roi=r) for r in rois]

        super(ExportMonitor, self).__init__(*args, **kwargs)

    @property
    def trajectories(self):
        return self._trajectories


    @property
    def first_frame(self):
        return self._first_frame


    def get_chunk_frame_range(self, chunk):
        return np.array(self._store._index.get_chunk_metadata(chunk)["frame_number"])[[0, -1]].tolist()


    def run(self, ncores=1):

        store_filename = os.path.join(self._store.filename, "metadata.yaml")
        chunks = self._chunks

        if ncores == 1:
            if self._frame_range is None:
                frame_range = self._frame_time_table["frame_number"].iloc[[0,-1]].values.tolist()
            else:
                frame_range = self._frame_range

            output = [self.run_single_thread(
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
            if self._frame_range is None:
                pass
            else:
                frame_ranges = [(e[0] + self._frame_range[0], e[0] + self._frame_range[1]) for e in frame_ranges]

            output = ProgressParallel(n_jobs=ncores, verbose=10)(
                delayed(self.run_single_thread)(
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
    def run_single_thread(trajectories, unit_trackers, output, frame_range, frame_time_table, store_filename, chunk=None, ncores=1):

        trajectories = trajectories[frame_range[0]:frame_range[1], :, :]
        output=get_output_filename(output, chunk)
        thread_safe_store = imgstore.new_for_filename(store_filename)

        rw = ResultWriterClass.from_trajectories(
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
