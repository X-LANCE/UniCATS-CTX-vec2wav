# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import logging
import os
import sys
import tarfile

from distutils.version import LooseVersion
from filelock import FileLock

import h5py
import numpy as np
import torch
import yaml


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


class HDF5ScpLoader(object):
    """Loader class for a fests.scp file of hdf5 file.

    Examples:
        key1 /some/path/a.h5:feats
        key2 /some/path/b.h5:feats
        key3 /some/path/c.h5:feats
        key4 /some/path/d.h5:feats
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        >>> array = loader["key1"]

        key1 /some/path/a.h5
        key2 /some/path/b.h5
        key3 /some/path/c.h5
        key4 /some/path/d.h5
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp", "feats")
        >>> array = loader["key1"]

        key1 /some/path/a.h5:feats_1,feats_2
        key2 /some/path/b.h5:feats_1,feats_2
        key3 /some/path/c.h5:feats_1,feats_2
        key4 /some/path/d.h5:feats_1,feats_2
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        # feats_1 and feats_2 will be concatenated
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp, default_hdf5_path="feats"):
        """Initialize HDF5 scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with hdf5 format.
            default_hdf5_path (str): Path in hdf5 file. If the scp contain the info, not used.

        """
        self.default_hdf5_path = default_hdf5_path
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get hdf5 file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        p = self.data[key]
        if ":" in p:
            if len(p.split(",")) == 1:
                return read_hdf5(*p.split(":"))
            else:
                p1, p2 = p.split(":")
                feats = [read_hdf5(p1, p) for p in p2.split(",")]
                return np.concatenate(
                    [f if len(f.shape) != 1 else f.reshape(-1, 1) for f in feats], 1
                )
        else:
            return read_hdf5(p, self.default_hdf5_path)

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


class NpyScpLoader(object):
    """Loader class for a fests.scp file of npy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...
        >>> loader = NpyScpLoader("feats.scp")
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp):
        """Initialize npy scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with npy format.

        """
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get npy file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        return np.load(self.data[key])

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


def load_model(checkpoint, config=None, stats=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import ctx_vec2wav.models

    # get model and load parameters
    model_class = getattr(
        ctx_vec2wav.models,
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    model = ctx_vec2wav.models.CTXVEC2WAVGenerator(
        ctx_vec2wav.models.CTXVEC2WAVFrontend(config["frontend_params"]['prompt_channels'], **config["frontend_params"]),
        model_class(**config["generator_params"])
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from ctx_vec2wav.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.backend.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model


def crop_seq(x, offsets, length):
    """Crop padded tensor with specified length.

    :param x: (torch.Tensor) The shape is (B, C, D)
    :param offsets: (list)
    :param min_len: (int)
    :return:
    """
    B, C, T = x.shape
    x_ = x.new_zeros(B, C, length)
    for i in range(B):
        x_[i, :] = x[i, :, offsets[i]: offsets[i] + length]
    return x_

