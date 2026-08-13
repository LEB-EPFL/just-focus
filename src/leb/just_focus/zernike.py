"""Zernike polynomial phase perturbations for the pupil, indexed by Noll's indices.

The public API takes only Noll's sequential indices (see `InputField.with_zernike_modes`),
following the definition at
https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices (equivalently,
OEIS A176988: https://oeis.org/A176988).

Zernike polynomial evaluation is delegated to the optional `zernipax
<https://github.com/PlasmaControl/ZERNIPAX>`_ dependency, which is not installed by
default. Install it with the `zernike` extra, e.g. `pip install just-focus[zernike]`,
to use `InputField.with_zernike_modes`.

"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

try:
    from zernipax.zernike import fourier, zernike_radial_cpu
except ImportError:
    fourier = None
    zernike_radial_cpu = None


class ZernipaxNotInstalledError(ImportError):
    """Raised when Zernike-mode functionality is used without the `zernike` extra."""

    def __init__(self) -> None:
        super().__init__(
            "Zernike polynomial support requires the optional 'zernipax' dependency. "
            "Install it with the 'zernike' extra, e.g. `pip install just-focus[zernike]`."
        )


def _noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll's sequential index j to Zernike radial degree n and azimuthal frequency m.

    Follows the definition of Noll's sequential indices given in [1]_ (equivalently,
    OEIS A176988 [2]_), inverted into a closed form: n is found from the triangular
    numbers that bound j, and m from the parity/sign rules that order modes within a
    given n. This is distinct from the ANSI/OSA single-index convention.

    Parameters
    ----------
    j : int
        Noll's sequential index. Must be a positive integer (j >= 1).

    Returns
    -------
    tuple of int
        The (n, m) pair corresponding to `j`.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices
    .. [2] https://oeis.org/A176988

    """
    if j < 1:
        raise ValueError(f"Noll index j must be a positive integer, got {j}.")

    # Each radial degree n contributes n + 1 modes; walk through degrees until the
    # one containing j is found.
    n = 0
    remainder = j - 1
    while remainder > n:
        n += 1
        remainder -= n

    m = (-1) ** j * ((n % 2) + 2 * ((remainder + (n + 1) % 2) // 2))
    return n, m


def zernike_pupil_coordinates(mesh_size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute polar pupil coordinates on the same normalized mesh used elsewhere.

    Parameters
    ----------
    mesh_size : int
        The size of the mesh grid for the pupil field.

    Returns
    -------
    tuple of NDArray[np.float64]
        The radial coordinate rho and azimuthal coordinate theta (radians), evaluated
        on the normalized pupil mesh where the pupil edge is at rho = 1.

    """
    normed_coords = np.linspace(-1, 1, mesh_size)
    px, py = np.meshgrid(normed_coords, normed_coords)
    rho = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return rho.astype(np.float64), theta.astype(np.float64)


@lru_cache(maxsize=32)
def _zernike_basis_cached(noll_indices: tuple[int, ...], mesh_size: int) -> NDArray[np.float64]:
    """Cache key-driven implementation backing `_zernike_basis`; see that function."""
    if zernike_radial_cpu is None or fourier is None:
        raise ZernipaxNotInstalledError()

    nm = np.array([_noll_to_nm(j) for j in noll_indices])
    n_vals, m_vals = nm[:, 0], nm[:, 1]

    rho, theta = zernike_pupil_coordinates(mesh_size)
    rho_flat = rho.ravel()
    theta_flat = theta.ravel()

    radial = np.asarray(zernike_radial_cpu(rho_flat, n_vals, m_vals))
    angular = np.asarray(fourier(theta_flat[:, np.newaxis], m_vals))

    # Noll normalization: unit RMS over the unit disk.
    normalization = np.where(m_vals == 0, np.sqrt(n_vals + 1), np.sqrt(2 * (n_vals + 1)))

    basis = (radial * angular * normalization).astype(np.float64)
    basis.setflags(write=False)  # shared across callers via the lru_cache
    return basis


def _zernike_basis(noll_indices: int | Sequence[int], mesh_size: int) -> NDArray[np.float64]:
    """Compute the fixed basis matrix for a set of Noll-normalized Zernike modes.

    This is the mode-geometry-dependent (and expensive, `zernipax`-backed) half of
    `InputField.with_zernike_modes`, split out so it can be cached and combined with
    coefficients separately (e.g. natively in the active array backend, so gradients
    can flow into the coefficients without differentiating through `zernipax`
    itself). Results are cached per `(noll_indices, mesh_size)`, since the basis
    does not depend on the coefficients and is safe to reuse across calls. Not part
    of the public API: callers only ever need `InputField.with_zernike_modes`.

    Returns a basis matrix of shape `(mesh_size**2, len(noll_indices))`, where
    column k is Noll-normalized Zernike mode `noll_indices[k]` flattened row-major
    over the normalized pupil mesh (matching `zernike_pupil_coordinates`). Each
    call returns an independent, writable copy — the underlying cache entry itself
    is never exposed, so mutating the result or converting it to another array
    backend (e.g. `torch.as_tensor`) cannot corrupt the cache.

    """
    noll_tuple = tuple(int(j) for j in np.atleast_1d(np.asarray(noll_indices, dtype=int)))
    return _zernike_basis_cached(noll_tuple, mesh_size).copy()
