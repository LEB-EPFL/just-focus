"""Dtype resolution for the active array backend and precision."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import backend
from .backend import Backend, Precision

try:
    import torch
except ImportError:
    torch = None

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[Any] | torch.Tensor
else:
    Array = Any


_NUMPY_FLOAT_DTYPES: dict[Precision, Any] = {
    Precision.FLOAT32: np.float32,
    Precision.FLOAT64: np.float64,
}
_NUMPY_COMPLEX_DTYPES: dict[Precision, Any] = {
    Precision.FLOAT32: np.complex64,
    Precision.FLOAT64: np.complex128,
}


def float_dtype() -> Any:
    """Return the real floating-point dtype for the active backend and precision."""
    precision = backend.get_precision()
    if backend.get_backend() is Backend.NUMPY:
        return _NUMPY_FLOAT_DTYPES[precision]
    return torch.float32 if precision is Precision.FLOAT32 else torch.float64


def complex_dtype() -> Any:
    """Return the complex dtype for the active backend and precision."""
    precision = backend.get_precision()
    if backend.get_backend() is Backend.NUMPY:
        return _NUMPY_COMPLEX_DTYPES[precision]
    return torch.complex64 if precision is Precision.FLOAT32 else torch.complex128
