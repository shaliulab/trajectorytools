import os.path
import json

from ethoscope.core.roi import ROI

from trajectorytools.trajectorytools import Trajectories

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


def get_commit_hash():
    hash_str = "v1.0"
    return hash_str
