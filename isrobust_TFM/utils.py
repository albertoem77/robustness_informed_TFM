import importlib.resources as pkg_resources
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def get_resource_path(fname):
    """Get path to pkg resources by filename.

    Returns
    -------
    pathlib.PosixPath
        Path to file.
    """
    with pkg_resources.path("ivae_scorer.resources", fname) as f:
        data_file_path = f

    return Path(data_file_path)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
