import os.path
import logging
import sys

from trajectorytools.trajectories import import_idtrackerai_dict
from .trajectories import Trajectories
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List

logger = logging.getLogger(__name__)

# Utils

def check_array_has_no_nans(arr):
    return not np.isnan(arr).any()


def _best_ids(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:

    # Input: two arrays of locations of the same shape
    number_of_individuals = xa.shape[0]
    try:

        assert xa.shape == (number_of_individuals, 2)
        assert xb.shape == (number_of_individuals, 2)
    except AssertionError as error:
        raise ValueError(
            "The number of individuals in the contiguous frames is not the same"
        )

    try:
        assert check_array_has_no_nans(xa)
        assert check_array_has_no_nans(xb)

    except AssertionError as error:
        raise ValueError(
            "The identity matching in the contiguous frames is ambiguous"
        )

    # We calculate the matrix of all distances between
    # all points in a and all points in b, and then find
    # the assignment that minimises the sum of distances
    distances = cdist(xa, xb)
    _, col_ind = linear_sum_assignment(distances)

    return col_ind


def get_last_index(arr):

    index = -1
    while not check_array_has_no_nans(arr[index,:]):
        index -= 1
    return index

def get_first_index(arr):

    index = 0
    while not check_array_has_no_nans(arr[index,:]):
        index += 1
    return index


def _concatenate_two_np(ta: np.ndarray, tb: np.ndarray, chunk_id=0):
    # Shape of ta, tb: (frames, individuals, 2)

    last_index = get_last_index(ta)
    first_index = get_first_index(tb)

    if last_index != -1 or first_index != 0:
        logger.warning(f"""Concatenation between chunks \
        {chunk_id}-{chunk_id+1} will use \
        {last_index} frame from left chunk and \
        {first_index} from right chunk
        """)

    try:
        best_ids = _best_ids(ta[last_index, :], tb[first_index, :])
    except ValueError as error:
        logger.error(
            f"Cannot concatenate frame {ta.shape[0]} and {ta.shape[0]+1}"
        )
        raise error

    return True, np.concatenate([ta, tb[:, best_ids, :]], axis=0)


def _concatenate_np(t_list: List[np.ndarray], zero_index=0) -> np.ndarray:

    if len(t_list) == 1:
        return (True, t_list[0])

    status, concatenation_until_now = _concatenate_np(t_list[:-1], zero_index=zero_index)

    last_concat_chunk = len(t_list[:-1])-1+zero_index

    if not status is True:
        return (status, concatenation_until_now)
    else:
        try:
            return _concatenate_two_np(concatenation_until_now, t_list[-1], chunk_id=last_concat_chunk)
        except Exception as error:
            logger.error(f"Concatenation error between 0-based chunks {last_concat_chunk} and {last_concat_chunk+1}")
            logger.error(error)
            return (last_concat_chunk, concatenation_until_now)


# Obtain trajectories from concatenation


def from_several_positions(t_list: List[np.ndarray], zero_index=0, **kwargs) -> Trajectories:
    """Obtains a single trajectory object from a concatenation
    of several arrays representing locations
    """
    status, t_concatenated = _concatenate_np(t_list, zero_index=zero_index)
    return Trajectories.from_positions(t_concatenated, **kwargs)


def _concatenate_idtrackerai_dicts(traj_dicts, **kwargs):
    """Concatenates several idtrackerai dictionaries.

    The output contains:
    - a concatenation of the trajectories
    - the values of the first diccionary for all other keys
    """
    traj_dict_cat = traj_dicts[0].copy()
    status, traj_cat = _concatenate_np(
        [traj_dict["trajectories"] for traj_dict in traj_dicts],
        **kwargs
    )
    traj_dict_cat["trajectories"] = traj_cat
    return status, traj_dict_cat


def _pick_trajectory_file(trajectories_folder, pref_index=-1):
    """
    Return the path to the last trajectory file in this folder
    based on the timestamp suffix added when
    pythonvideoannotator_module_idtrackerai.models.video.objects.
    idtrackerai_object_io.IdtrackeraiObjectIO.save_updated_identities

    is run.

    The original file without the timestamp, produced by idtrackerai alone,
    will be selected last by default
    """
    trajectory_files = sorted(
        [f for f in os.listdir(trajectories_folder)],
        key=lambda x: os.path.splitext(x)[0],
    )
    return os.path.join(trajectories_folder, trajectory_files[pref_index])


def pick_w_wo_gaps(session_folder, allow_human=True):
    """Select the best trajectories file
    available in an idtrackerai session
    """
    trajectories_wo_gaps = os.path.join(session_folder, "trajectories_wo_gaps")
    trajectories = os.path.join(session_folder, "trajectories")

    if allow_human:
        pref_index=-1
    else:
        pref_index=0

    if os.path.exists(trajectories_wo_gaps):
        return _pick_trajectory_file(trajectories_wo_gaps, pref_index)
    elif os.path.exists(trajectories):
        return _pick_trajectory_file(trajectories, pref_index)
    else:
        raise Exception(f"Session {session_folder} has no trajectories")


def is_idtrackerai_session(path):
    """Check whether the passed path is an idtrackerai session"""
    return os.path.exists(os.path.join(path, "video_object.npy"))


def get_trajectories(idtrackerai_collection_folder, *args, **kwargs):
    """Return a list of all trajectory files available in an idtrackerai collection folder"""
    file_contents = os.listdir(idtrackerai_collection_folder)

    file_contents = [
        os.path.join(idtrackerai_collection_folder, folder)
        for folder in file_contents
    ]

    idtrackerai_sessions = []
    for folder in file_contents:
        if is_idtrackerai_session(folder):
            idtrackerai_sessions.append(folder)

    trajectories_paths = {
        os.path.basename(session): pick_w_wo_gaps(session, *args, **kwargs)
        for session in idtrackerai_sessions
    }
    trajectories_paths = {
        k: v for k, v in trajectories_paths.items() if not v is None
    }
    return trajectories_paths


def from_several_idtracker_files(
    trajectories_paths, zero_index=0, **kwargs
):

    traj_dicts = []

    for trajectories_path in trajectories_paths:
        traj_dict = np.load(
            trajectories_path, encoding="latin1", allow_pickle=True
        ).item()
        traj_dicts.append(traj_dict)

    status, traj_dict = _concatenate_idtrackerai_dicts(traj_dicts, zero_index=zero_index)
    tr = import_idtrackerai_dict(traj_dict, **kwargs)
    tr.params["path"] = trajectories_paths
    tr.params["construct_method"] = "from_several_idtracker_files"
    return status, tr


def diagnose_concatenation(trajectories_paths):

    problematic_junctions = []
    zero_index = 0
    while True:
        last_concat, _ =from_several_idtracker_files(trajectories_paths[zero_index:], zero_index=zero_index)
        if last_concat is True:
            break

        problematic_junctions.append(last_concat)
        zero_index = last_concat + 1
        print(zero_index)

    return problematic_junctions
