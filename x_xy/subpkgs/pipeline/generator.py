from typing import Optional, Tuple

import jax

import x_xy
from x_xy.algorithms import (
    Generator,
    Normalizer,
    RCMG_Config,
    make_normalizer_from_generator,
)
from x_xy.base import System
from x_xy.subpkgs import pipeline


def _to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def make_generator(
    configs: RCMG_Config | list[RCMG_Config],
    bs: int,
    sys_data: System | list[System],
    sys_noimu: Optional[System] = None,
    imu_attachment: Optional[dict] = None,
    return_xs: bool = False,
    normalize: bool = False,
    randomize_positions: bool = True,
    random_s2s_ori: bool = False,
    noisy_imus: bool = True,
    quasi_physical: bool | list[bool] = False,
    smoothen_degree: Optional[int] = None,
) -> Tuple[Generator, Optional[Normalizer]]:
    normalizer = None
    if normalize:
        gen, _ = make_generator(
            configs,
            bs,
            sys_data,
            sys_noimu,
            imu_attachment,
            return_xs,
            False,
            randomize_positions,
            random_s2s_ori,
            noisy_imus,
            quasi_physical,
            smoothen_degree,
        )
        normalizer = make_normalizer_from_generator(gen)

    configs, sys_data = _to_list(configs), _to_list(sys_data)

    if sys_noimu is None:
        sys_noimu, _imu_attachment = pipeline.make_sys_noimu(sys_data[0])
        if imu_attachment is None:
            imu_attachment = _imu_attachment

    assert sys_noimu is not None
    assert imu_attachment is not None

    def _make_generator(sys, config, quasi_physical: bool):
        def finalize_fn(key, q, x, sys):
            X = pipeline.imu_data(
                key,
                x,
                sys,
                imu_attachment,
                noisy=noisy_imus,
                random_s2s_ori=random_s2s_ori,
                quasi_physical=quasi_physical,
                smoothen_degree=smoothen_degree if not quasi_physical else None,
            )
            y = x_xy.algorithms.rel_pose(sys_noimu, x, sys)

            if normalizer is not None:
                X = normalizer(X)

            if return_xs:
                return X, y, x
            else:
                return X, y

        def setup_fn(key, sys):
            if randomize_positions:
                key, consume = jax.random.split(key)
                sys = x_xy.algorithms.setup_fn_randomize_positions(consume, sys)
            # this just randomizes the joint axes, this random joint-axes
            # is only used if the joint type is `rr`
            # this is why there is no boolean `randomize_jointaxes` argument
            key, consume = jax.random.split(key)
            sys = x_xy.algorithms.setup_fn_randomize_joint_axes(consume, sys)
            return sys

        return x_xy.algorithms.build_generator(
            sys,
            config,
            setup_fn,
            finalize_fn,
        )

    gens = []
    for sys in sys_data:
        for config in configs:
            for qp in _to_list(quasi_physical):
                gens.append(_make_generator(sys, config, qp))

    assert (bs // len(gens)) > 0, f"Batchsize too small. Must be at least {len(gens)}"
    batchsizes = len(gens) * [bs // len(gens)]
    return x_xy.algorithms.batch_generator(gens, batchsizes), normalizer
