import os

import jax
import jax.numpy as jnp

import x_xy


def is_pytest():
    return "PYTEST_CURRENT_TEST" in os.environ


def test_animate():
    dt = 1e-2
    filename = "animation"
    xml = "test_double_pendulum"
    sys = x_xy.io.load_example(xml)
    sys = sys.replace(dt=dt)

    q = jnp.array([0, 1.0])
    qd = jnp.zeros((sys.qd_size(),))

    state = x_xy.base.State.create(sys, q, qd)

    if is_pytest():
        T = 1
    else:
        T = 10

    step_fn = jax.jit(x_xy.step)

    xs = []
    for _ in range(int(T / sys.dt)):
        state = step_fn(sys, state, jnp.zeros_like(state.qd))
        xs.append(state.x)

    fmts = ["mp4"]

    if is_pytest():
        fmts += ["gif"]

    for fmt in fmts:
        x_xy.animate(filename, sys, xs, fmt=fmt)

        if is_pytest():
            os.system(f"rm animation.{fmt}")


def test_shapes():
    sys_str = """
<x_xy model="shape_test">
    <options gravity="0 0 9.81" dt="0.01" />
    <worldbody>
        <geom type="sphere" mass="1" pos="0 0 0" dim="0.3" color="white" />
        <geom type="box" mass="1" pos="-1 0 0" quat="1 0 1 0" dim="1 0.3 0.2" color="0.8 0.3 1 0" />
        <geom type="cylinder" mass="1" pos="1 0 0.5" quat="0.75 0 0 0.25" dim="0.3 1" color="0.2 0.8 0.5" />
        <geom type="capsule" mass="1" pos="0 0 -1" dim="0.3 2" />

        <body name="dummy" pos="0 0 0" quat="1 0 0 0" joint="ry" />
    </worldbody>
</x_xy>
    """  # noqa: E501

    sys = x_xy.load_sys_from_str(sys_str)
    state = x_xy.State.create(sys)
    step_fn = jax.jit(x_xy.step)
    state = step_fn(sys, state)
    x_xy.animate("figures/example.png", sys, state.x, fmt="png")


if __name__ == "__main__":
    test_animate()