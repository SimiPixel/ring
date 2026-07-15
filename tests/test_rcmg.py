import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ring
from ring.maths import unit_quats_like
from ring.maths import wrap_to_pi


@pytest.mark.parametrize("N,seed", [(1, 0), (1, 1), (5, 0), (5, 1)])
def test_batch_generator(N: int, seed: int):
    sys = ring.io.load_example("test_free")
    config1 = ring.MotionConfig(ang0_min=0.0, ang0_max=0.0)
    config2 = ring.MotionConfig()
    gen1 = ring.RCMG(
        sys, config1, finalize_fn=lambda key, q, x, sys: (q, x)
    ).to_lazy_gen()
    gen2 = ring.RCMG(
        sys, config2, finalize_fn=lambda key, q, x, sys: (q, x)
    ).to_lazy_gen()
    gen = ring.algorithms.generator.batch.generators_lazy(
        [gen1, gen2, gen1], [N, N, N], jit=0
    )
    q, _ = gen(jax.random.PRNGKey(seed))

    arr_eq = lambda a, b: np.testing.assert_allclose(
        q[a:b, 0, :4], unit_quats_like(q[a:b, 0, :4])
    )

    arr_eq(0, N)
    with pytest.raises(AssertionError):
        arr_eq(N, 2 * N)
    arr_eq(2 * N, 3 * N)


def test_initial_ang_pos_values():
    T = 1.0
    bs = 8
    pos_min, pos_max = -5.0, 5.0
    # system consists only of prismatic, and then revolute joint
    sys = ring.io.load_example("test_ang0_pos0")

    def rcmg(ang0_min=0, ang0_max=0, pos0_min=0, pos0_max=0, bs=bs):
        gen = ring.RCMG(
            sys,
            ring.MotionConfig(
                ang0_min=ang0_min,
                ang0_max=ang0_max,
                pos0_min=pos0_min,
                pos0_max=pos0_max,
                pos_min=pos_min,
                pos_max=pos_max,
                T=T,
            ),
            finalize_fn=lambda key, q, x, sys: (q, x),
        ).to_lazy_gen(bs, jit=False)
        q, _ = gen(jax.random.PRNGKey(1))
        return q

    for init_val in np.linspace(pos_min, pos_max, num=10):
        q = rcmg(init_val, init_val, init_val, init_val)
        np.testing.assert_allclose(q[:, 0, 0], init_val * jnp.ones((bs,)))
        np.testing.assert_allclose(q[:, 0, 1], wrap_to_pi(init_val * jnp.ones((bs,))))

    a, b = -0.1, 0.1
    c, d = -1.5, -0.5
    q = rcmg(a, b, c, d)[:, 0].T
    assert np.all((q[1] > a) & (q[1] < b))
    assert np.all((q[0] > c) & (q[0] < d))


def _dang_max(t: jax.Array) -> jax.Array:
    return jnp.where(t < 0.5, 1.0, 2.0)


def test_rcmg():
    for example in ["test_all_1"]:
        sys = ring.io.load_example(example)
        for cdf_bins_min, cdf_bins_max in zip([1, 1, 3], [1, 3, 3]):
            config = ring.MotionConfig(
                T=1.0,
                cdf_bins_min=cdf_bins_min,
                cdf_bins_max=cdf_bins_max,
                randomized_interpolation_angle=True,
                # this tests `TimeDependentFloat`-logic
                dang_max=_dang_max,
            )

            bs = 8

            generator = ring.RCMG(
                sys, config, finalize_fn=lambda key, q, x, sys: (q, x)
            ).to_lazy_gen(bs, jit=False)

            qs, xs = generator(
                jax.random.PRNGKey(
                    1,
                )
            )

            assert qs.shape == (bs, 100, sys.q_size())


def test_correlated_trajectory_fn():
    system = ring.io.load_example("test_free")
    config = ring.MotionConfig(T=0.1)

    def trajectory_fn(key, sys, motion_config, sample_count):
        del key
        samples = (
            int(motion_config.T / sys.dt)
            if sample_count is None
            else sample_count
        )
        q = jnp.zeros((samples, sys.q_size()))
        return q.at[:, 0].set(1.0)

    generator = ring.RCMG(
        system,
        config,
        trajectory_fn=trajectory_fn,
        finalize_fn=lambda key, q, x, sys: (q, x.pos),
    ).to_lazy_gen(sizes=2, jit=False)
    q, positions = generator(jax.random.PRNGKey(2))

    assert q.shape == (2, 10, system.q_size())
    np.testing.assert_allclose(q[..., 0], 1.0)
    np.testing.assert_allclose(q[..., 1:], 0.0)
    assert positions.shape == (2, 10, system.num_links(), 3)


def test_endpoint_constrained_trajectory_is_jittable_and_preserves_center():
    system = ring.io.load_example("test_control")
    config = ring.MotionConfig(T=0.1)

    def base(key, sys, motion_config, sample_count):
        samples = (
            int(motion_config.T / sys.dt)
            if sample_count is None
            else sample_count
        )
        q = jnp.broadcast_to(ring.State.create(sys).q, (samples, sys.q_size()))
        q = q.at[:, 4].set(jnp.linspace(0.0, 0.1, samples))
        angle = jax.random.uniform(key, minval=-jnp.pi, maxval=jnp.pi)
        return q.at[:, 7].set(angle).at[:, 8].set(-0.5 * angle)

    trajectory = ring.EndpointConstrainedTrajectory(
        base=base,
        endpoints=(
            ring.LinkPoint("6D"),
            ring.LinkPoint("lower", (1.0, 0.0, 0.0)),
        ),
        minimum_distance=1.5,
        attempts=30,
        center_points=(
            ring.LinkPoint("upper", (0.5, 0.0, 0.0)),
            ring.LinkPoint("lower", (0.5, 0.0, 0.0)),
        ),
        preserve_root_trajectory_as_center=True,
    )
    q = jax.jit(lambda key: trajectory(key, system, config, None))(
        jax.random.PRNGKey(1)
    )
    transforms, _ = jax.vmap(
        ring.algorithms.forward_kinematics_transforms, (None, 0)
    )(system, q)

    def point(name, offset):
        index = system.name_to_idx(name)
        offsets = jnp.broadcast_to(jnp.array(offset), (len(q), 3))
        return transforms.pos[:, index] + ring.maths.rotate(
            offsets, ring.maths.quat_inv(transforms.rot[:, index])
        )

    distance = jnp.linalg.norm(
        point("lower", (1.0, 0.0, 0.0)) - point("6D", (0.0, 0.0, 0.0)),
        axis=1,
    )
    center = 0.5 * (
        point("upper", (0.5, 0.0, 0.0))
        + point("lower", (0.5, 0.0, 0.0))
    )
    assert np.percentile(distance, 1) >= 1.5
    np.testing.assert_allclose(center[:, 0], np.linspace(0.0, 0.1, len(q)), atol=1e-5)


def test_rcmg_sizes_arg():
    s = ring.io.load_example("test_double_pendulum")
    c = ring.MotionConfig(T=3.0)
    make = lambda n_s, n_c: ring.RCMG(
        n_s * [s], n_c * [c], add_y_relpose=1, disable_tqdm=1
    )

    # size too small
    with pytest.raises(AssertionError):
        make(1, 2).to_list(1)

    with pytest.raises(AssertionError):
        make(2, 1).to_list(1)

    with pytest.raises(AssertionError):
        make(2, 3).to_list(5)

    with pytest.raises(AssertionError):
        make(2, 3).to_list([1, 6])

    with pytest.raises(AssertionError):
        make(2, 3).to_list([2, 6])

    # split uniform not possible
    with pytest.raises(AssertionError):
        make(2, 3).to_list(9)

    with pytest.raises(AssertionError):
        make(3, 2).to_list(8)

    d = make(2, 3).to_list([3, 6])
    assert len(d) == 9

    d = make(2, 3).to_list([6, 3])
    assert len(d) == 9

    # n_gens != n_sizes
    with pytest.raises(AssertionError):
        make(3, 2).to_list([3, 6])

    d = make(3, 2).to_list([2, 2, 4])
    assert len(d) == 8

    d = make(3, 2).to_list(6)
    assert len(d) == 6
