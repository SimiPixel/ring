import jax
import jax.numpy as jnp

from x_xy import base, maths


def accelerometer(xs: base.Transform, gravity: jax.Array, dt: float) -> jax.Array:
    """Compute measurements of an accelerometer that follows a frame which moves along
    a trajectory of Transforms. Let `xs` be the transforms from base to link.
    """
    N = xs.pos.shape[0]
    acc = jnp.zeros((N, 3))
    acc = acc.at[1:-1].set((xs.pos[:-2] + xs.pos[2:] - 2 * xs.pos[1:-1]) / dt**2)

    if isinstance(gravity, float) or gravity.shape == ():
        gravity = jnp.array([0.0, 0, gravity])

    # gravity is a scalar value
    acc = acc + gravity

    return maths.rotate(acc, xs.rot)


def gyroscope(rot: jax.Array, dt: float) -> jax.Array:
    """Compute measurements of a gyroscope that follows a frame with an orientation
    given by trajectory of quaternions `rot`."""
    # this was not needed before
    # the reason is that before q represented
    # FROM LOCAL TO EPS
    # but now we q represents
    # FROM EPS TO LOCAL
    q = maths.quat_inv(rot)

    q = jnp.vstack((q, jnp.array([[1.0, 0, 0, 0]])))
    # 1st-order approx to derivative
    dq = maths.quat_mul(maths.quat_inv(q[:-1]), q[1:])

    axis, angle = maths.quat_to_rot_axis(dq)
    angle = angle[:, None]

    gyr = axis * angle / dt
    return jnp.where(jnp.abs(angle) > 1e-10, gyr, jnp.zeros(3))


NOISE_LEVELS = {"acc": 0.5, "gyr": jnp.deg2rad(1.0)}
BIAS_LEVELS = {"acc": 0.5, "gyr": jnp.deg2rad(1.0)}


def add_noise_bias(key: jax.random.PRNGKey, imu_measurements: dict) -> dict:
    """Add noise and bias to 6D imu measurements.

    Args:
        key (jax.random.PRNGKey): Random seed.
        imu_measurements (dict): IMU measurements without noise and bias.
            Format is {"gyr": Array, "acc": Array}.

    Returns:
        dict: IMU measurements with noise and bias.
    """
    noisy_imu_measurements = {}
    for sensor in ["acc", "gyr"]:
        key, c1, c2 = jax.random.split(key)
        noise = (
            jax.random.normal(c1, shape=imu_measurements[sensor].shape)
            * NOISE_LEVELS[sensor]
        )
        bias = jax.random.uniform(
            c2, minval=-BIAS_LEVELS[sensor], maxval=BIAS_LEVELS[sensor]
        )
        noisy_imu_measurements[sensor] = imu_measurements[sensor] + noise + bias
    return noisy_imu_measurements
