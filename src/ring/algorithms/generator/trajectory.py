"""Composable whole-system trajectory policies for RCMG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from ring import base
from ring import maths
from ring.algorithms.kinematics import forward_kinematics_transforms
from ring.algorithms.generator.types import TRAJECTORY_FN


@dataclass(frozen=True)
class LinkPoint:
    """A physical point expressed in a link's local coordinates."""

    link_name: str
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class EndpointConstrainedTrajectory:
    """Reject trajectories that collapse two physical endpoint points.

    The accepted trajectory can additionally move its free-root translation so
    that the physical chain center follows the base trajectory's root-link
    position. This correction is written into ``q`` before RCMG computes final
    forward kinematics.
    """

    base: TRAJECTORY_FN
    endpoints: tuple[LinkPoint, LinkPoint]
    minimum_distance: float
    minimum_median_distance: Optional[float] = None
    distance_quantile: float = 0.01
    attempts: int = 30
    center_points: tuple[LinkPoint, ...] = ()
    preserve_root_trajectory_as_center: bool = False

    def __post_init__(self) -> None:
        if len(self.endpoints) != 2:
            raise ValueError("exactly two endpoints are required")
        if self.minimum_distance < 0.0:
            raise ValueError("minimum_distance must be non-negative")
        if (
            self.minimum_median_distance is not None
            and self.minimum_median_distance < 0.0
        ):
            raise ValueError("minimum_median_distance must be non-negative")
        if not 0.0 <= self.distance_quantile <= 1.0:
            raise ValueError("distance_quantile must be between zero and one")
        if self.attempts < 1:
            raise ValueError("attempts must be positive")
        if self.preserve_root_trajectory_as_center and not self.center_points:
            raise ValueError("center_points are required for center preservation")

    @staticmethod
    def _point_positions(
        transforms: base.Transform,
        system: base.System,
        point: LinkPoint,
    ) -> jax.Array:
        index = system.name_to_idx(point.link_name)
        rotation = transforms.rot[:, index]
        offset = jnp.broadcast_to(jnp.asarray(point.offset), (len(rotation), 3))
        return transforms.pos[:, index] + maths.rotate(
            offset, maths.quat_inv(rotation)
        )

    def _score(self, q: jax.Array, system: base.System) -> jax.Array:
        transforms, _ = jax.vmap(
            forward_kinematics_transforms, (None, 0)
        )(system, q)
        first = self._point_positions(transforms, system, self.endpoints[0])
        second = self._point_positions(transforms, system, self.endpoints[1])
        distance = jnp.linalg.norm(second - first, axis=1)
        low_score = jnp.quantile(distance, self.distance_quantile) / max(
            self.minimum_distance, 1e-8
        )
        if self.minimum_median_distance is None:
            return low_score
        median_score = jnp.median(distance) / max(
            self.minimum_median_distance, 1e-8
        )
        return jnp.minimum(low_score, median_score)

    def _preserve_center(self, q: jax.Array, system: base.System) -> jax.Array:
        if not self.preserve_root_trajectory_as_center:
            return q
        root_name = system.find_body_to_world(name=True)
        root_q = system.idx_map("q")[root_name]
        if root_q.stop - root_q.start < 7:
            raise ValueError("center preservation requires a free or COR root")
        transforms, _ = jax.vmap(
            forward_kinematics_transforms, (None, 0)
        )(system, q)
        root_index = system.name_to_idx(root_name)
        desired_center = transforms.pos[:, root_index]
        center = jnp.mean(
            jnp.stack(
                [
                    self._point_positions(transforms, system, point)
                    for point in self.center_points
                ],
                axis=1,
            ),
            axis=1,
        )
        world_delta = desired_center - center
        root_rotation = jnp.broadcast_to(
            system.links.transform1.rot[root_index], (len(q), 4)
        )
        root_delta = maths.rotate(world_delta, root_rotation)
        return q.at[:, root_q.start + 4 : root_q.start + 7].add(
            root_delta
        )

    def __call__(
        self,
        key: jax.Array,
        system: base.System,
        motion_config: object,
        sample_count: Optional[int],
    ) -> jax.Array:
        keys = jax.random.split(key, self.attempts)
        first = self.base(keys[0], system, motion_config, sample_count)
        first_score = self._score(first, system)
        first_valid = first_score >= 1.0

        def choose(carry, candidate_key):
            best_score, best, accepted, selected = carry

            def evaluate(key):
                candidate = self.base(
                    key, system, motion_config, sample_count
                )
                return candidate, self._score(candidate, system)

            candidate, score = jax.lax.cond(
                accepted,
                lambda _: (selected, best_score),
                evaluate,
                candidate_key,
            )
            valid = score >= 1.0
            better = score > best_score
            best = jnp.where(better, candidate, best)
            best_score = jnp.where(better, score, best_score)
            choose_candidate = jnp.logical_and(jnp.logical_not(accepted), valid)
            selected = jnp.where(choose_candidate, candidate, selected)
            accepted = jnp.logical_or(accepted, valid)
            return (best_score, best, accepted, selected), None

        initial = (first_score, first, first_valid, first)
        if self.attempts > 1:
            (best_score, best, accepted, selected), _ = jax.lax.scan(
                choose, initial, keys[1:]
            )
            del best_score
            q = jnp.where(accepted, selected, best)
        else:
            q = first
        return self._preserve_center(q, system)
