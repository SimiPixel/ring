from typing import Any, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import tree_utils as tu
from flax import struct
from jax.tree_util import tree_map

from x_xy import maths

Scalar = jax.Array
Vector = jax.Array
Quaternion = jax.Array


class _Base:
    """Base functionality of all spatial datatypes.
    Copied and modified from https://github.com/google/brax/blob/main/brax/v2/base.py
    """

    def __add__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: Any) -> Any:
        return tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        return tree_map(lambda x: x / o, self)

    def __getitem__(self, i: int) -> Any:
        return self.take(i)

    def reshape(self, shape: Sequence[int]) -> Any:
        return tree_map(lambda x: x.reshape(shape), self)

    def slice(self, beg: int, end: int) -> Any:
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return tree_map(lambda x: jnp.take(x, i, axis=axis), self)

    def hstack(self, *others: Any) -> Any:
        return tree_map(lambda *x: jnp.hstack(x), self, *others)

    def vstack(self, *others: Any) -> Any:
        return tree_map(lambda *x: jnp.vstack(x), self, *others)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def batch(self, *others, along_existing_first_axis: bool = False) -> Any:
        return tu.tree_batch((self,) + others, along_existing_first_axis, "jax")

    def index_set(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].add(y), self, o)

    @property
    def T(self):
        return tree_map(lambda x: x.T, self)

    def flatten(self, num_batch_dims: int = 0) -> jax.Array:
        return tu.batch_concat(self, num_batch_dims)

    def squeeze(self):
        return tree_map(lambda x: jnp.squeeze(x), self)

    def squeeze_1d(self):
        return tree_map(lambda x: jnp.atleast_1d(jnp.squeeze(x)), self)

    def batch_dim(self) -> int:
        return tu.tree_shape(self)


@struct.dataclass
class Transform(_Base):
    """Represents the Transformation from Plücker A to Plücker B,
    where B is located relative to A at `pos` in frame A and `rot` is the
    relative quaternion from A to B."""

    pos: Vector
    rot: Quaternion

    @classmethod
    def create(cls, pos=None, rot=None):
        assert not (pos is None and rot is None), "One must be given."
        if pos is None:
            pos = jnp.zeros((3,))
        if rot is None:
            rot = jnp.array([1.0, 0, 0, 0])
        return Transform(pos, rot)

    @classmethod
    def zero(cls, shape=()) -> "Transform":
        """Returns a zero transform with a batch shape."""
        pos = jnp.zeros(shape + (3,))
        rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
        return Transform(pos, rot)

    def as_matrix(self) -> jax.Array:
        E = maths.quat_to_3x3(self.rot)
        return maths.spatial.quadrants(aa=E, bb=E) @ maths.spatial.xlt(self.pos)


@struct.dataclass
class Motion(_Base):
    "Coordinate vector that represents a spatial motion vector in Plücker Coordinates."
    ang: Vector
    vel: Vector

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Motion(ang, vel)

    @classmethod
    def zero(cls, shape=()) -> "Motion":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Motion(ang, vel)

    def as_matrix(self):
        return self.flatten()


@struct.dataclass
class Force(_Base):
    "Coordinate vector that represents a spatial force vector in Plücker Coordinates."
    ang: Vector
    vel: Vector

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Force(ang, vel)

    @classmethod
    def zero(cls, shape=()) -> "Force":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Force(ang, vel)

    def as_matrix(self):
        return self.flatten()


@struct.dataclass
class Inertia(_Base):
    """Spatial Inertia Matrix in Plücker Coordinates.
    Note that `h` is *not* the center of mass."""

    it_3x3: jax.Array
    h: Vector
    mass: Vector

    @classmethod
    def create(cls, mass: Vector, CoM: Vector, it_3x3: jnp.ndarray):
        """Construct spatial inertia of an object with mass `mass` located
        at the center of mass `CoM` and an inertia matrix `it_3x3` around that
        center of mass.
        """
        it_3x3 = maths.spatial.mcI(mass, CoM, it_3x3)[:3, :3]
        h = mass * CoM
        return cls(it_3x3, h, mass)

    @classmethod
    def zero(cls, shape=()) -> "Inertia":
        it_shape_3x3 = jnp.zeros(shape + (3, 3))
        h = jnp.zeros(shape + (3,))
        mass = jnp.zeros(shape + (1,))
        return cls(it_shape_3x3, h, mass)

    def as_matrix(self):
        hcross = maths.spatial.cross(self.h)
        return maths.spatial.quadrants(
            self.it_3x3, hcross, -hcross, self.mass * jnp.eye(3)
        )


@struct.dataclass
class Geometry(_Base):
    mass: jax.Array
    CoM: jax.Array


@struct.dataclass
class Sphere(Geometry):
    radius: jax.Array
    vispy_kwargs: dict = struct.field(False, default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = 2 / 5 * self.mass * self.radius**2 * jnp.eye(3)
        return it_3x3


@struct.dataclass
class Box(Geometry):
    dim_x: jax.Array
    dim_y: jax.Array
    dim_z: jax.Array
    vispy_kwargs: dict = struct.field(False, default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.array(
                [
                    [self.dim_y**2 + self.dim_z**2, 0, 0],
                    [0, self.dim_x**2 + self.dim_z**2, 0],
                    [0, 0, self.dim_x**2 + self.dim_y**2],
                ]
            )
        )
        return it_3x3

    @classmethod
    def cube(cls, mass, CoM, dim, vispy_kwargs={}):
        return cls(mass, CoM, dim, dim, dim, vispy_kwargs)

    @classmethod
    def standard_segment(cls, vispy_kwargs={}):
        dim_x = 0.4
        dim_y = 0.2
        dim_z = 0.1
        return cls(
            5.0,
            jnp.array([dim_x / 2, dim_y / 2, dim_z / 2]),
            dim_x,
            dim_y,
            dim_z,
            vispy_kwargs,
        )


@struct.dataclass
class Cylinder(Geometry):
    """Length is along x-axis."""

    radius: jax.Array
    length: jax.Array
    vispy_kwargs: dict = struct.field(False, default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        radius_dir = 3 * self.radius**2 + self.length**2
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.array(
                [
                    [6 * self.radius**2, 0, 0],
                    [0, radius_dir, 0],
                    [0, 0, radius_dir],
                ]
            )
        )
        return it_3x3


N_JOINT_PARAMS: int = 1


@struct.dataclass
class Link(_Base):
    transform1: Transform

    # these parameters can be used to model joints that have parameters
    # they are directly feed into the `jcalc` routine
    # this array *must* be of shape (N_JOINT_PARAMS,)
    joint_params: jax.Array = jnp.zeros((N_JOINT_PARAMS,))

    # internal useage
    inertia: Inertia = Inertia.zero()
    transform2: Transform = Transform.zero()
    transform: Transform = Transform.zero()


Q_WIDTHS = {
    "free": 7,
    "frozen": 0,
    "spherical": 4,
    "px": 1,
    "py": 1,
    "pz": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
}
QD_WIDTHS = {
    "free": 6,
    "frozen": 0,
    "spherical": 3,
    "px": 1,
    "py": 1,
    "pz": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
}


@struct.dataclass
class System(_Base):
    link_parents: list[int] = struct.field(False)
    links: Link
    link_types: list[str] = struct.field(False)
    link_damping: jax.Array
    link_armature: jax.Array
    # simulation timestep size
    dt: float = struct.field(False)
    # whether or not to re-calculate the inertia
    # matrix at every simulation timestep because
    # the geometries may have changed
    dynamic_geometries: bool = struct.field(False)
    # geometries in the system
    # len(geoms) == len(links)
    geoms: list[list[Geometry]]
    # root / base acceleration offset
    gravity: jax.Array = jnp.array([0, 0, -9.81])

    integration_method: str = struct.field(
        False, default_factory=lambda: "semi_implicit_euler"
    )
    mass_mat_iters: int = struct.field(False, default_factory=lambda: 0)

    link_names: list[str] = struct.field(False, default_factory=lambda: [])

    model_name: Optional[str] = struct.field(False, default_factory=lambda: None)

    def num_links(self) -> int:
        return len(self.link_parents)

    def q_size(self) -> int:
        return sum([Q_WIDTHS[typ] for typ in self.link_types])

    def qd_size(self) -> int:
        return sum([QD_WIDTHS[typ] for typ in self.link_types])

    def name_to_idx(self, name: str) -> int:
        return self.link_names.index(name)


@struct.dataclass
class State(_Base):
    q: jax.Array
    qd: jax.Array
    x: Transform
    mass_mat_inv: jax.Array

    @classmethod
    def create(cls, sys: System, q=None, qd=None):
        if q is None:
            q = jnp.zeros((sys.q_size(),))
        if qd is None:
            qd = jnp.zeros((sys.qd_size(),))
        x = Motion.zero((sys.num_links(),))
        return cls(q, qd, x, jnp.diag(jnp.ones((sys.qd_size(),))))
