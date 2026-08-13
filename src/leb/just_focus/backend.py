"""Array backend selection: run this package on NumPy or PyTorch.

Public API: `set_backend("torch")` (or `"numpy"`, the default), optionally
with `precision="float32"`, chooses the backend used to build new arrays.
This is process-wide state: it affects `Pupil`/`InputField` objects
constructed *after* the call, not ones already built — an existing object
keeps the backend it was built with, and mixing a NumPy-built `Pupil` with
a PyTorch-built `InputField` fails at the first elementwise operation that
combines an `ndarray` with a `Tensor`.

`be` is an internal implementation detail, not part of the public API. This
package's own array-construction code (`inputs.py`, `pupil.py`,
`focal_fields.py`) uses it in place of `np`/`torch` directly: call functions
on it the way you'd call them on `numpy`, e.g. `be.exp(x)`, `be.linspace(0,
1, 10)`, and it resolves to whichever backend is currently active. It's
safe for that internal use because everything those modules build stays
within a single `set_backend()` call.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class Backend(StrEnum):
    """The array backend used to construct and propagate fields.

    Attributes
    ----------
    NUMPY : str
        Use NumPy arrays. The default; requires no optional dependencies.
    TORCH : str
        Use PyTorch tensors. Requires the `torch` extra, e.g.
        `pip install just-focus[torch]`.
    """
    NUMPY = "numpy"
    TORCH = "torch"


class Precision(StrEnum):
    """Floating-point precision used to construct arrays/tensors.

    Attributes
    ----------
    FLOAT32 : str
        Single precision (float32 / complex64).
    FLOAT64 : str
        Double precision (float64 / complex128). Default.
    """
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class TorchNotInstalledError(ImportError):
    """Raised when the torch backend is requested without the `torch` extra."""

    def __init__(self) -> None:
        super().__init__(
            "PyTorch backend support requires the optional 'torch' dependency. "
            "Install it with the 'torch' extra, e.g. `pip install just-focus[torch]`."
        )


_backend: Backend = Backend.NUMPY
_precision: Precision = Precision.FLOAT64


def set_backend(
    name: Backend | str = Backend.NUMPY,
    *,
    precision: Precision | str = Precision.FLOAT64,
) -> None:
    """Set the active array backend and floating-point precision.

    Affects only arrays/tensors constructed *after* this call; objects
    already built (e.g. an existing `Pupil` or `InputField`) keep whatever
    backend was active when they were constructed.

    Parameters
    ----------
    name : Backend or str
        The backend to activate, e.g. `Backend.TORCH` or `"torch"`.
    precision : Precision or str, optional
        The floating-point precision to activate. Default is double
        precision (float64/complex128).

    Raises
    ------
    TorchNotInstalledError
        If `name` is `Backend.TORCH` but PyTorch is not installed.
    """
    global _backend, _precision

    backend = Backend(name)
    if backend is Backend.TORCH and torch is None:
        raise TorchNotInstalledError()

    _backend = backend
    _precision = Precision(precision)


def get_backend() -> Backend:
    """Return the currently active backend."""
    return _backend


def get_precision() -> Precision:
    """Return the currently active floating-point precision."""
    return _precision


def _active_module() -> Any:
    """Return the top-level module (`numpy` or `torch`) for the active backend."""
    return np if _backend is Backend.NUMPY else torch


class _BackendProxy:
    """Singleton proxy exposing the active backend's array namespace.

    Generic operations (`linspace`, `exp`, `sqrt`, `meshgrid`, the `fft`
    submodule, ...) are delegated via `__getattr__` to whichever module is
    currently active, resolved fresh on every access. A handful of methods
    are implemented explicitly because NumPy and PyTorch diverge in ways a
    plain attribute lookup can't bridge (padding argument order, in-place
    vs. functional dtype casts, host-array conversion, ...).
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_active_module(), name)

    def pad(
        self,
        arr: Any,
        pad_width: tuple[tuple[int, int], tuple[int, int]],
        value: float = 0.0,
    ) -> Any:
        """Zero-pad a 2D array/tensor by `pad_width` (NumPy's nested per-axis form)."""
        if _backend is Backend.NUMPY:
            return np.pad(arr, pad_width, mode="constant", constant_values=value)
        # torch.nn.functional.pad wants a flat tuple, last dimension first.
        flat_pad = (pad_width[1][0], pad_width[1][1], pad_width[0][0], pad_width[0][1])
        return torch.nn.functional.pad(arr, flat_pad, mode="constant", value=value)

    def astype(self, arr: Any, dtype: Any) -> Any:
        """Cast `arr` to `dtype`."""
        if _backend is Backend.NUMPY:
            return np.astype(arr, dtype)
        return arr.to(dtype)

    def copy(self, arr: Any) -> Any:
        """Return an independent copy of `arr`."""
        if _backend is Backend.NUMPY:
            return np.copy(arr)
        return arr.clone()

    def to_numpy(self, arr: Any) -> np.ndarray:
        """Return `arr` as a host NumPy array, regardless of the active backend."""
        if _backend is Backend.NUMPY:
            return np.asarray(arr)
        return arr.detach().cpu().numpy()

    def asarray(self, data: Any, dtype: Any = None) -> Any:
        """Construct an array/tensor of the active backend from `data`."""
        if _backend is Backend.NUMPY:
            return np.asarray(data, dtype=dtype)
        return torch.as_tensor(data, dtype=dtype)

    @property
    def float_dtype(self) -> Any:
        """The active backend's real floating-point dtype for the active precision."""
        from . import dtypes
        return dtypes.float_dtype()

    @property
    def complex_dtype(self) -> Any:
        """The active backend's complex dtype for the active precision."""
        from . import dtypes
        return dtypes.complex_dtype()


be = _BackendProxy()
