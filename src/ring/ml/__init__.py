from . import base
from . import callbacks
from . import ml_utils
from . import optimizer
from . import ringnet
from . import train
from . import training_loop
from .base import AbstractFilter
from .ml_utils import on_cluster
from .ml_utils import unique_id
from .optimizer import make_optimizer
from .ringnet import RING
from .train import train_fn


def RING_ICML24(params=None, eval: bool = True, **kwargs):
    """Create the RING network used in the icml24 paper.

    X[..., :3]  = acc
    X[..., 3:6] = gyr
    X[..., 6:9] = jointaxis
    X[..., 9:]  = dt
    """
    from pathlib import Path

    if params is None:
        params = Path(__file__).parent.joinpath("params/0x13e3518065c21cd8.pickle")

    ringnet = RING(params=params, **kwargs)  # noqa: F811
    ringnet = base.ScaleX_FilterWrapper(ringnet)
    if eval:
        ringnet = base.LPF_FilterWrapper(ringnet, 10.0, samp_freq=None)
    ringnet = base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet


def RNNO(
    output_dim: int,
    return_quats: bool = False,
    params=None,
    eval: bool = True,
    **kwargs,
):
    assert "message_dim" not in kwargs
    assert "link_output_normalize" not in kwargs
    assert "link_output_dim" not in kwargs

    ringnet = RING(  # noqa: F811
        params=params,
        message_dim=0,
        link_output_normalize=False,
        link_output_dim=output_dim,
        **kwargs,
    )
    ringnet = base.NoGraph_FilterWrapper(ringnet, quat_normalize=return_quats)
    ringnet = base.ScaleX_FilterWrapper(ringnet)
    if eval and return_quats:
        ringnet = base.LPF_FilterWrapper(ringnet, 10.0, samp_freq=None)
    if return_quats:
        ringnet = base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet
