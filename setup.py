import fnmatch
import os

import setuptools


def find_data_files(package_dir, patterns, excludes=()):
    """Recursively finds files whose names match the given shell patterns."""
    paths = set()

    def is_excluded(s):
        for exclude in excludes:
            if fnmatch.fnmatch(s, exclude):
                return True
        return False

    for directory, _, filenames in os.walk(package_dir):
        if is_excluded(directory):
            continue
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                # NB: paths must be relative to the package directory.
                relative_dirpath = os.path.relpath(directory, package_dir)
                full_path = os.path.join(relative_dirpath, filename)
                if not is_excluded(full_path):
                    paths.add(full_path)
    return list(paths)


setuptools.setup(
    name="x_xy",
    packages=setuptools.find_packages(),
    version="0.7.0",
    package_data={
        "x_xy": find_data_files(
            "x_xy", patterns=["*.xml", "*.yaml", "*.joblib", "*.json"]
        ),
    },
    include_package_data=True,
    install_requires=[
        "jaxlib",
        "jax",
        "flax",
        "tqdm",
        "vispy",
        "imageio[ffmpeg]==2.27",
        "pytest",
        "tree_utils @ git+https://github.com/SimiPixel/tree_utils.git",
        "joblib",
        "pyyaml",
    ],
    entry_points={"console_scripts": ["xxy-render = x_xy.cli.render:main"]},
)


# TODO
# create a separate `dev` package that additionally requires the following
# dependencies; install via pip
# - mkdocs
# - mkdocs-material
# - mkdocstrings
# - mkdocstrings-python
# - mknotebooks
