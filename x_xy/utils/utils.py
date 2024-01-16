import io
from pathlib import Path
import shutil
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import wget

from x_xy.base import _Base
from x_xy.base import Geometry


def tree_equal(a, b):
    "Copied from Marcel / Thomas"
    if type(a) is not type(b):
        return False
    if isinstance(a, _Base):
        return tree_equal(a.__dict__, b.__dict__)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(tree_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(tree_equal(a[i], b[i]) for i in range(len(a)))
    if isinstance(a, (jax.Array, np.ndarray)):
        return jnp.allclose(a, b)
    return a == b


def _sys_compare_unsafe(sys1, sys2, verbose: bool, prefix: str) -> bool:
    d1 = sys1.__dict__
    d2 = sys2.__dict__
    for key in d1:
        if isinstance(d1[key], _Base):
            if not _sys_compare_unsafe(d1[key], d2[key], verbose, prefix + "." + key):
                return False
        elif isinstance(d1[key], list) and isinstance(d1[key][0], Geometry):
            for ele1, ele2 in zip(d1[key], d2[key]):
                if not _sys_compare_unsafe(ele1, ele2, verbose, prefix + "." + key):
                    return False
        else:
            if not tree_equal(d1[key], d2[key]):
                if verbose:
                    print(f"Systems different in attribute `sys{prefix}.{key}`")
                    print(f"{repr(d1[key])} NOT EQUAL {repr(d2[key])}")
                return False
    return True


def sys_compare(sys1, sys2, verbose: bool = True):
    equalA = _sys_compare_unsafe(sys1, sys2, verbose, "")
    equalB = tree_equal(sys1, sys2)
    assert equalA == equalB
    return equalA


def to_list(obj: object) -> list:
    "obj -> [obj], if it isn't already a list."
    if not isinstance(obj, list):
        return [obj]
    return obj


def dict_union(
    d1: dict[str, jax.Array] | dict[str, dict[str, jax.Array]],
    d2: dict[str, jax.Array] | dict[str, dict[str, jax.Array]],
    overwrite: bool = False,
) -> dict:
    "Builds the union between two nested dictonaries."
    # safety copying; otherwise this function would mutate out of scope
    d1 = pytree_deepcopy(d1)
    d2 = pytree_deepcopy(d2)

    for key2 in d2:
        if key2 not in d1:
            d1[key2] = d2[key2]
        else:
            if not isinstance(d2[key2], dict) or not isinstance(d1[key2], dict):
                raise Exception(f"d1.keys()={d1.keys()}; d2.keys()={d2.keys()}")

            for key_nested in d2[key2]:
                if not overwrite:
                    assert (
                        key_nested not in d1[key2]
                    ), f"d1.keys()={d1[key2].keys()}; d2.keys()={d2[key2].keys()}"

            d1[key2].update(d2[key2])
    return d1


def dict_to_nested(
    d: dict[str, jax.Array], add_key: str
) -> dict[str, dict[str, jax.Array]]:
    "Nests a dictonary by inserting a single key dictonary."
    return {key: {add_key: d[key]} for key in d.keys()}


_xxy_cache_foldername = ".xxy_cache"


def download_from_repo(path_in_repo: str, repo: str = "x_xy_v2_datahost") -> str:
    "Download file from `x_xy_v2` Github repo. Returns path on disk."
    path_on_disk = (
        Path("~").expanduser().joinpath(_xxy_cache_foldername).joinpath(path_in_repo)
    )
    if not path_on_disk.exists():
        path_on_disk.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/SimiPixel/{repo}/main/{path_in_repo}"
        print(f"Downloading file from url {url}.. (this might take a moment)")
        wget.download(url, out=str(path_on_disk.parent))
        print(
            f"Downloading finished. Saved to location {path_on_disk}. "
            "All downloaded files can be deleted with "
            "`x_xy.utils.delete_download_cache`."
        )
    return str(path_on_disk)


def delete_download_cache(only: Optional[str] = None) -> None:
    "Delete folder and all content in `~/.xxy_cache`."
    path_cache_folder = Path("~").expanduser().joinpath(_xxy_cache_foldername)
    if only is not None:
        path_cache_folder = path_cache_folder.joinpath(only)

    if Path(path_cache_folder).exists():
        shutil.rmtree(path_cache_folder)


def save_figure_to_rgba(fig) -> np.ndarray:
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def pytree_deepcopy(tree):
    "Recursivley copies a pytree with numpy/jax array leafs."
    if isinstance(tree, jax.Array):
        return tree
    elif isinstance(tree, np.ndarray):
        return tree.copy()
    elif isinstance(tree, list):
        return [pytree_deepcopy(ele) for ele in tree]
    elif isinstance(tree, tuple):
        return tuple(pytree_deepcopy(ele) for ele in tree)
    elif isinstance(tree, dict):
        return {key: pytree_deepcopy(value) for key, value in tree.items()}
    else:
        raise NotImplementedError(f"Not implemented for type={type(tree)}")
