import importlib.util

import numpy as np
import pytest

from leb.just_focus import backend
from leb.just_focus.backend import Backend, Precision, TorchNotInstalledError, be
from leb.just_focus.dtypes import complex_dtype, float_dtype

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch


def test_default_backend_is_numpy():
    assert backend.get_backend() is Backend.NUMPY
    assert backend.get_precision() is Precision.FLOAT64


def test_set_backend_accepts_strings():
    backend.set_backend("numpy", precision="float32")
    assert backend.get_backend() is Backend.NUMPY
    assert backend.get_precision() is Precision.FLOAT32


@pytest.mark.skipif(HAS_TORCH, reason="torch is installed")
def test_set_backend_torch_raises_when_not_installed():
    with pytest.raises(TorchNotInstalledError):
        backend.set_backend(Backend.TORCH)
    # A failed set_backend call must not mutate the active backend.
    assert backend.get_backend() is Backend.NUMPY


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_set_backend_torch_when_installed():
    backend.set_backend(Backend.TORCH)
    assert backend.get_backend() is Backend.TORCH


def test_float_dtype_numpy_default():
    assert float_dtype() == np.float64
    assert complex_dtype() == np.complex128


def test_float_dtype_numpy_single_precision():
    backend.set_backend(Backend.NUMPY, precision=Precision.FLOAT32)
    assert float_dtype() == np.float32
    assert complex_dtype() == np.complex64


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_float_dtype_torch():
    backend.set_backend(Backend.TORCH)
    assert float_dtype() == torch.float64
    assert complex_dtype() == torch.complex128

    backend.set_backend(Backend.TORCH, precision=Precision.FLOAT32)
    assert float_dtype() == torch.float32
    assert complex_dtype() == torch.complex64


def test_be_identity_is_stable_across_set_backend():
    proxy_before = be
    backend.set_backend(Backend.NUMPY, precision=Precision.FLOAT32)
    assert be is proxy_before


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_be_delegates_to_active_backend():
    backend.set_backend(Backend.NUMPY)
    assert isinstance(be.linspace(0, 1, 4), np.ndarray)

    backend.set_backend(Backend.TORCH)
    assert isinstance(be.linspace(0, 1, 4), torch.Tensor)


def test_be_pad_matches_numpy_convention():
    arr = np.ones((2, 3))
    padded = be.pad(arr, ((1, 1), (2, 2)))
    assert padded.shape == (4, 7)
    assert padded.sum() == arr.sum()
