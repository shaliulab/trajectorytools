import tempfile
import json
import datetime
import logging
import traceback
logger = logging.getLogger(__name__)

from joblib import Parallel, delayed
import tqdm
import numpy as np
import imgstore
import pandas as pd

from trajectorytools import Trajectories


from trajectorytools.export.monitor import ExportMonitor
from trajectorytools.export.io import EthoscopeExport
from trajectorytools.export.parallel import ProgressParallel, delayed


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
