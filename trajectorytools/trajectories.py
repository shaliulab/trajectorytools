import logging
logger = logging.getLogger(__name__)
import warnings
from copy import deepcopy

import numpy as np

import trajectorytools as tt
from trajectorytools.fish_bouts import find_bouts_individual


def calculate_center_of_mass(trajectories, params):
    """calculate_center_of_mass

    Produces a CenterMassTrajectory, with the position, velocity and acceleration
    of the center of mass.

    :param trajectories: Dictionary of numpy arrays for position ('_s'),
    velocity ('_v') and acceleration ('_a')
    :param params: Dictionary of parameters
    """
    center_of_mass = {
        k: np.mean(trajectories[k], axis=1) for k in Trajectory.keys_to_copy
    }
    return CenterMassTrajectory(center_of_mass, params)


def estimate_center_and_radius(locations):
    """Estimates center and radius of the smallest circle containing all points

    :param locations: Numpy array of locations. It can be any shape, but last
    dim must be 2 (x, y)
    """
    center_x, center_y, estimated_radius = tt.find_enclosing_circle(locations)
    center_a = np.array([center_x, center_y])
    return center_a, estimated_radius


def radius_and_center_from_traj_dict(locations, traj_dict):
    """Obtains radius and center of the arena

    If the trajectories contain the arena radius/center information, use it
    Otherwise, return None to estimate radius/center from trajectories

    :param locations: Numpy array of locations. Last dim must be (x, y)
    :param traj_dict:
    """
    if "setup_points" in traj_dict and "border" in traj_dict["setup_points"]:
        center_a, radius = estimate_center_and_radius(
            traj_dict["setup_points"]["border"]
        )
    else:
        arena_radius = traj_dict.get("arena_radius", None)
        arena_center = traj_dict.get("arena_center", None)

        # Find center and radius. Then override if necessary
        if arena_radius is None or arena_center is None:
            estimated_center, estimated_radius = estimate_center_and_radius(
                locations
            )

        if arena_radius is None:
            radius = estimated_radius
        else:
            radius = arena_radius

        if arena_center is None:
            center_a = estimated_center
        else:
            center_a = arena_center

    return radius, center_a


class Trajectory:
    keys_to_copy = ["_s", "_v", "_a"]
    own_params = True

    def __init__(self, trajectories, params):
        for key in self.keys_to_copy:
            setattr(self, key, trajectories[key])
        if self.own_params:
            params = deepcopy(params)
        self.params = params

    def __len__(self):
        return len(self._s)

    # Properties and methods with no side-effects
    # i.e. they do not change class member parameters

    def point_from_px(self, point):
        return (point + self.params["displacement"]) / self.params[
            "length_unit"
        ]

    def point_to_px(self, point):
        return point * self.params["length_unit"] - self.params["displacement"]

    def vector_from_px(self, vector):
        return vector / self.params["length_unit"]

    def vector_to_px(self, vector):
        return vector * self.params["length_unit"]

    @property
    def number_of_frames(self):
        return self._s.shape[0]

    @property
    def s(self):
        return self.point_from_px(self._s)

    @property
    def v(self):
        return self.vector_from_px(self._v) * self.params["time_unit"]

    @property
    def a(self):
        return self.vector_from_px(self._a) * (self.params["time_unit"] ** 2)

    @property
    def speed(self):
        return tt.norm(self.v)

    @property
    def acceleration(self):
        return tt.norm(self.a)

    @property
    def e(self):
        return tt.normalise(self.v)

    @property
    def tg_acceleration(self):
        return tt.dot(self.a, self.e)

    @property
    def curvature(self):
        return tt.curvature(self.v, self.a)

    @property
    def normal_acceleration(self):
        return np.square(self.speed) * self.curvature

    @property
    def distance_travelled(self):
        return tt.geometry.distance_travelled(self.s)

    @property
    def straightness(self):
        return tt.geometry.straightness(self.s)

    def estimate_center_and_radius_from_locations(self, in_px=False):
        """Assumes that the trajectories are restricted to a circular area and
        estimates its center and radius from the trajectories

        :param in_px: If True, the results are given in the original
        frame of reference and scale (usually pixels).
        """
        if not in_px:
            center_a, estimated_radius = estimate_center_and_radius(self.s)
        else:
            center_a, estimated_radius = estimate_center_and_radius(self._s)
        return center_a, estimated_radius

    # Properties with side-effects
    # i.e. they change class member parameters

    def new_length_unit(self, length_unit, length_unit_name="?"):
        factor = self.params["length_unit"] / length_unit
        if self.own_params:
            if "radius" in self.params:
                self.params["radius"] = self.params["radius"] * factor
            self.params["length_unit"] = length_unit
            self.params["length_unit_name"] = length_unit_name
        return factor  # In the future it will return self

    def new_time_unit(self, time_unit, time_unit_name="?"):
        factor = self.params["time_unit"] / time_unit
        if self.own_params:
            self.params["time_unit"] = time_unit
            self.params["time_unit_name"] = time_unit_name
        return factor  # In the future it will return self

    def origin_to(self, new_origin):
        """Places origin of frame of reference in a given location

        :param new_origin: Point that will become our new origin.
        It is expressed in the original frame of reference (usually px).
        """
        if self.own_params:
            self.params["displacement"] = -new_origin
        return self

    def resample(self, new_frame_rate, **kwargs):
        # This function modifies _s, _v and _a
        # In the future it will not modify the current Trajectory
        # But it will produce a new one
        if "frame_rate" not in self.params:
            raise Exception("Frame rate not in trajectories")
        old_frame_rate = self.params["frame_rate"]
        fraction = new_frame_rate / old_frame_rate
        self._s = tt.resample(self._s, new_frame_rate, old_frame_rate, kwargs)
        self._v = tt.resample(self._v, new_frame_rate, old_frame_rate, kwargs)
        self._a = tt.resample(self._a, new_frame_rate, old_frame_rate, kwargs)
        if self.own_params:
            self.params["frame_rate"] = new_frame_rate
            self.params["time_unit"] = self.params["time_unit"] * fraction
        return self

    # methods and properties wrt points
    def distance_to(self, point):
        return tt.norm(self.s - point)

    def _projection_vector_towards(self, point, vector):
        return tt.dot(tt.normalise(point - self.s), vector)

    def orientation_towards(self, point):
        warnings.warn(Warning("Deprecated. Use angle_towards instead"))
        return self.angle_towards(point)

    def angle_towards(self, point):
        return np.arccos(np.clip(self.e_towards(point), -1, 1))

    def signed_angle_towards(self, point):
        return tt.signed_angle_between_vectors(point - self.s, self.v)

    def e_towards(self, point):
        return self._projection_vector_towards(point, self.e)

    def speed_towards(self, point):
        return self._projection_vector_towards(point, self.v)

    def acceleration_towards(self, point):
        return self._projection_vector_towards(point, self.a)

    @property
    def distance_to_center(self):
        raise Exception(
            "Deprecated: Center trajectories with origin_to and use distance_to_origin"
        )

    @property
    def distance_to_origin(self):
        return self.distance_to(np.zeros(2))


class CenterMassTrajectory(Trajectory):
    own_params = False  # Parameters are shared with parent


class Trajectories(Trajectory):
    def __init__(self, trajectories, params):
        super().__init__(trajectories, params)
        self.center_of_mass = calculate_center_of_mass(
            trajectories, self.params
        )


    def cartesian(self, arrays, out=None):
        """
        https://stackoverflow.com/a/28684982/3541756

        Generate a cartesian product of input arrays.

        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

        """

        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype

        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        #m = n / arrays[0].size
        m = int(n / arrays[0].size)
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
            #for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
        return out



    def extend(self, other):
        """
        Extend a trajectory using the data from the consecutive trajectory

        Identities are propagated by assuming the position of the same animal on the last frame of the oldest
        trajectory and the the first of the newest should be very similar i.e.
        the animal that is closest on the new frame to a given animal on the old frame is the same animal

        This may have to be proofread in extreme cases
        where the animals move too fast or the framerate is too low
        """
        last_seen = self._s[-1,:,:]
        next_seen = other._s[0,:, :]


        dist=np.sqrt(
            np.diff(self.cartesian([last_seen[:,0], next_seen[:,0]]), axis=1)**2 +
            np.diff(self.cartesian([last_seen[:,1], next_seen[:,1]]), axis=1)**2
        )
        dist=dist.reshape((last_seen.shape[0], next_seen.shape[0]))

        correct_id=dist.argmin(axis=1)

        for key in self.keys_to_copy:
            logger.info(f"Extending {key} key")
            self.__dict__[key]=np.concatenate([
                self.__dict__[key],
                other.__dict__[key][:, correct_id, :]
            ])

        # #viz
        # dist /= (dist.max() / 255)
        # plt.set_cmap("gray")
        # plt.imshow(np.uint8(np.round(dist)))

    def _dict_to_save(self):
        traj_data = {key: self.__dict__[key] for key in self.keys_to_copy}
        params = self.params
        return {
            "traj_data": traj_data,
            "params": params,
            "class_name": self.__class__.__name__,
        }

    def save(self, filename):
        np.save(filename, self._dict_to_save())

    @classmethod
    def load(cls, filename):
        loaded_dict = np.load(filename, allow_pickle=True).item()
        return cls(loaded_dict["traj_data"], loaded_dict["params"])

    def export_trajectories_to_csv(self, csvfilename, delimiter=",", **kwargs):
        """Export trajectories to a csv plain text file.

        :param csvfilename: Name of the csv file to be created
        :param delimiter: Delimiter (default is ",")
        :param kwargs: Other keyword arguments for numpy.savetxt
        """
        data_to_save = np.concatenate([self.s, self.v, self.a], axis=-1)
        data_to_save = data_to_save.reshape(self.number_of_frames, -1)
        data_header = ["x", "y", "vx", "vy", "ax", "ay"]
        individuals = range(self.number_of_individuals)
        header = delimiter.join(
            [f"{h}_{i}" for i in individuals for h in data_header]
        )
        np.savetxt(
            csvfilename,
            data_to_save,
            header=header,
            delimiter=delimiter,
            **kwargs,
        )

    def __getitem__(self, val):
        view_trajectories = {
            k: getattr(self, k)[val] for k in self.keys_to_copy
        }
        return self.__class__(view_trajectories, self.params)

    def restrict_individuals(self, val):
        view_trajectories = {
            k: getattr(self, k)[:, val] for k in self.keys_to_copy
        }
        return self.__class__(view_trajectories, self.params)

    def __str__(self):
        if "path" in self.params:
            maybe_loaded = f"from {self.params['path']} "
        else:
            maybe_loaded = ""
        return (
            f"<{self.__class__.__name__} "
            + maybe_loaded
            + f"-- frames:{self._s.shape[0]}, individuals:{self._s.shape[1]}>"
        )

    @classmethod
    def from_idtracker(cls, trajectories_path, **kwargs):
        warnings.warn(Warning("Deprecated. Use from_idtrackerai instead"))
        return cls.from_idtrackerai(trajectories_path, **kwargs)

    @classmethod
    def from_idtrackerai(cls, trajectories_path, *args, **kwargs):
        """Create Trajectories from a idtracker.ai trajectories file

        :param trajectories_path: idtracker.ai generated file
        """
        traj_dict = np.load(
            trajectories_path, encoding="latin1", allow_pickle=True
        ).item()
        tr = cls.from_idtracker_(traj_dict, *args, **kwargs)
        tr.params["path"] = trajectories_path
        tr.params["construct_method"] = "from_idtrackerai"
        return tr

    @classmethod
    def from_idtracker_(
        cls,
        traj_dict,
        *args,
        interpolate_nans=True,
        center=False,
        smooth_params=None,
        dtype=np.float64,
        **kwargs
    ):
        """Create Trajectories from a idtracker.ai trajectories dictionary

        :param traj_dict: idtracker.ai generated dictionary
        :param interpolate_nans: whether to interpolate NaNs
        :param center: Whether to center trajectories, using a center estimated
        from the trajectories.
        :param smooth_params: Parameters of smoothing
        :param dtype: Desired dtype of trajectories.
        """

        t = traj_dict["trajectories"].astype(dtype)
        traj = cls.from_positions(
            t, *args, interpolate_nans=interpolate_nans, smooth_params=smooth_params, **kwargs
        )

        radius, center_ = radius_and_center_from_traj_dict(traj._s, traj_dict)
        traj.params.update(
            dict(radius=radius, radius_px=radius, _center=center_)
        )
        if center:
            traj.origin_to(traj.params["_center"])

        traj.params["frame_rate"] = traj_dict.get("frames_per_second", None)
        traj.params["body_length_px"] = traj_dict.get("body_length", None)
        traj.params["construct_method"] = "from_idtracker_"
        return traj

    @classmethod
    def from_positions(cls, t, *args, interpolate_nans=True, smooth_params=None,**kwargs):
        """Trajectory from positions

        :param t: Positions nd.array.
        :param interpolate_nans: whether to interpolate NaNs
        :param smooth_params: Arguments for smoothing (see tt.smooth)
        """
        if smooth_params is None:
            smooth_params = {"sigma": -1, "only_past": False}
        # Interpolate trajectories
        if interpolate_nans:
            tt.interpolate_nans(t)

        displacement = np.array([0.0, 0.0])

        # Smooth trajectories
        if smooth_params["sigma"] > 0:
            t_smooth = tt.smooth(t, **smooth_params)
        else:
            t_smooth = t

        trajectories = {}

        if smooth_params.get("only_past", False):
            [
                trajectories["_s"],
                trajectories["_v"],
                trajectories["_a"],
            ] = tt.velocity_acceleration_backwards(t_smooth, *args, **kwargs)
        else:
            [
                trajectories["_s"],
                trajectories["_v"],
                trajectories["_a"],
            ] = tt.velocity_acceleration(t_smooth, *args, **kwargs)

        # TODO: Organise the params dictionary more hierarchically
        # Maybe in the future add a "how_construct" key being a dictionary
        # with the classmethod called, path, interpolate_nans,
        # and smooth_params
        params = {
            "displacement": displacement,  # Units: pixels
            "length_unit": 1,  # Units: pixels
            "length_unit_name": "px",
            "time_unit": 1,  # In frames
            "time_unit_name": "frames",
            "interpolate_nans": interpolate_nans,
            "smooth_params": smooth_params,
            "construct_method": "from_positions",
        }

        return cls(trajectories, params)

    def normalise_by(self, normaliser):
        if not isinstance(normaliser, str):
            raise Exception(
                "normalise_by needs a string. To normalise by"
                "a number, please use new_length_unit"
            )
        if normaliser == "body length" or normaliser == "body_length":
            length_unit = self.params["body_length_px"]
            length_unit_name = "BL"
        elif normaliser == "radius":
            length_unit = self.params["radius_px"]
            length_unit_name = "R"
        else:
            raise Exception("Unknown key")
        self.new_length_unit(length_unit, length_unit_name)
        return self

    # Methods with side effects
    def resample(self, *args, **kwargs):
        self.center_of_mass.resample(*args, **kwargs)
        super().resample(*args, **kwargs)

    # Properties (without side effects)

    @property
    def interindividual_distances(self):
        return tt.socialcontext.interindividual_distances(self.s)

    @property
    def mean_interindividual_distances(self):
        return np.nansum(self.interindividual_distances, axis=-1) / (
            self.number_of_individuals - 1
        )

    @property
    def number_of_individuals(self):
        return self._s.shape[1]

    @property
    def identity_labels(self):
        # Placeholder, in case in the future labels are explicitly given
        return np.arange(self.number_of_individuals)


class FishTrajectories(Trajectories):
    def get_bouts(self, find_max_dict=None, find_min_dict=None):
        """Obtain bouts start and peak for all individuals

        :param find_max_dict: named arguments passed to scipy.signal.find_peaks
        :param find_min_dict: named arguments passed to scipy.signal.find_peaks

        returns
        :all_bouts: list of arrays, one array per individual. The dimensions
        of every array are (number_of_bouts, 3). Col-1 is the starting frame
        of the bout, col-2 is the peak of the bout, col-3 is the beginning of
        the next bout
        """

        all_bouts = []
        speed = self.speed
        for focal in range(self.number_of_individuals):
            bouts = find_bouts_individual(speed[:, focal])
            all_bouts.append(bouts)
        return all_bouts


class TrajectoriesWithPoints(Trajectories):
    def __init__(self, trajectories, params, points=None):
        super().__init__(trajectories, params)
        self._points = points if points is not None else {}

    def _dict_to_save(self):
        update_dict = {"points": self.points}
        return {**super()._dict_to_save(), **update_dict}

    @classmethod
    def load(cls, filename):
        loaded_dict = np.load(filename, allow_pickle=True).item()
        return cls(
            loaded_dict["traj_data"],
            loaded_dict["params"],
            points=loaded_dict["points"],
        )

    @classmethod
    def from_idtracker_(cls, traj_dict, **kwargs):
        twp = super().from_idtracker_(traj_dict, **kwargs)
        twp._points = traj_dict["setup_points"]
        return twp

    def __getitem__(self, val):
        view_traj_with_points = super().__getitem__(val)
        view_traj_with_points._points = self._points
        return view_traj_with_points

    @property
    def points(self):
        # TODO: only transform the keys needed, for efficiency
        return self.points_from_px(self._points)

    def points_from_px(self, points):
        new_points = {}
        for key in points:
            new_points[key] = self.point_from_px(points[key])
        return new_points

    def distance_to_point(self, key):
        return self.distance_to(self.points[key])

    def orientation_towards_point(self, key):
        return self.orientation_towards(self.points[key])

    def angle_towards_point(self, key):
        return self.angle_towards(self.points[key])

    def signed_angle_towards_point(self, key):
        return self.signed_angle_towards(self.points[key])

    def e_towards_point(self, key):
        return self.e_towards(self.points[key])

    def speed_towards_point(self, key):
        return self.speed_towards(self.points[key])

    def acceleration_towards_point(self, key):
        return self.acceleration_towards(self.points[key])
