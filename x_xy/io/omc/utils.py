import os
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import tree_utils
from vispy.scene import TurntableCamera

import x_xy


def autodetermine_imu_freq(path_imu) -> int:
    hz = []
    for file in os.listdir(path_imu):
        file = Path(path_imu).joinpath(file)
        if file.suffix != ".txt":
            continue

        with open(file) as f:
            f.readline()
            # second line in txt file is: // Update Rate: 40.0Hz
            second_line = f.readline()
            before = len("// Update Rate:")
            hz.append(int(float(second_line[before:-3])))

    assert len(set(hz)) == 1, f"IMUs have multiple sampling rates {hz}"
    return hz[0]


def autodetermine_optitrack_freq(path_optitrack):
    def find_framerate_in_line(line: str, key: str):
        before = line.find(key) + len(key) + 1
        return int(float(line[before:].split(",")[0]))

    # first line is:
    # ...,Capture Frame Rate,120.000000,Export Frame Rate,120.000000,...
    with open(path_optitrack) as f:
        line = f.readline()
        hz_cap = find_framerate_in_line(line, "Capture Frame Rate")
        hz_exp = find_framerate_in_line(line, "Export Frame Rate")
        assert hz_cap == hz_exp, "Capture and exported frame rate are not equal"

    return hz_exp


def state_trajectory_omc(
    sys: x_xy.base.System,
    omc_data: dict,
    t1: float = 0.0,
    t2: Optional[float] = None,
    eps_frame: Optional[str] = None,
):
    if t2 is None:
        t2i = tree_utils.tree_shape(omc_data)
    else:
        t2i = int(t2 / sys.dt)
    t1i = int(t1 / sys.dt)
    omc_data = jax.tree_map(lambda arr: jnp.array(arr)[t1i:t2i], omc_data)

    transforms = []

    if eps_frame is not None:
        eps = omc_data[eps_frame]
        q_eps = eps["quat"][0]
        q_eps = x_xy.maths.quat_inv(q_eps)
        t_eps = x_xy.base.Transform(eps["pos"][0], q_eps)
    else:
        t_eps = x_xy.base.Transform.zero()

    def f(_, __, link_name: str):
        q, pos = omc_data[link_name]["quat"], omc_data[link_name]["pos"]
        q = x_xy.maths.quat_inv(q)
        t = x_xy.base.Transform(pos, q)
        t = x_xy.algebra.transform_mul(t, x_xy.algebra.transform_inv(t_eps))
        transforms.append(t)

    x_xy.scan.tree(sys, f, "l", sys.link_names)

    transforms = transforms[0].batch(*transforms[1:])
    transforms = transforms.transpose((1, 0, 2))

    return transforms


def render_omc(
    sys_xml: str,
    omc_data: dict,
    elevation=30,
    distance=3,
    azimuth=5,
    filename: Optional[str] = None,
    **kwargs,
):
    sys = x_xy.io.load_sys_from_xml(sys_xml)
    transforms = state_trajectory_omc(sys, omc_data, **kwargs)

    camera = TurntableCamera(elevation=elevation, distance=distance, azimuth=azimuth)
    if filename is not None:
        x_xy.render.animate(filename, sys, transforms, camera=camera, show_cs=True)
    else:
        x_xy.render.gui(sys, transforms, camera=camera, show_cs=True)