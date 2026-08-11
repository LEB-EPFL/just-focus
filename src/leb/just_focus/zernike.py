"""Zernike polynomial phase perturbations for the pupil, indexed by Noll's indices.

The public API takes only Noll's sequential indices (see `zernike_phase`), following
the definition at https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices
(equivalently, OEIS A176988: https://oeis.org/A176988).

Zernike polynomial evaluation is delegated to the optional `zernipax
<https://github.com/PlasmaControl/ZERNIPAX>`_ dependency, which is not installed by
default. Install it with the `zernike` extra, e.g. `pip install just-focus[zernike]`,
to use `zernike_phase` or `InputField.with_zernike_modes`.

"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .dtypes import Float

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


def zernike_pupil_coordinates(mesh_size: int) -> tuple[NDArray[Float], NDArray[Float]]:
    """Compute polar pupil coordinates on the same normalized mesh used elsewhere.

    Parameters
    ----------
    mesh_size : int
        The size of the mesh grid for the pupil field.

    Returns
    -------
    tuple of NDArray[Float]
        The radial coordinate rho and azimuthal coordinate theta (radians), evaluated
        on the normalized pupil mesh where the pupil edge is at rho = 1.

    """
    normed_coords = np.linspace(-1, 1, mesh_size)
    px, py = np.meshgrid(normed_coords, normed_coords)
    rho = np.sqrt(px**2 + py**2)
    theta = np.arctan2(py, px)
    return rho.astype(Float), theta.astype(Float)


def zernike_phase(
    noll_indices: int | Sequence[int],
    coefficients: float | Sequence[float],
    mesh_size: int,
) -> NDArray[Float]:
    """Compute a pupil phase from a weighted sum of Noll-normalized Zernike polynomials.

    Requires the `zernike` extra.

    Parameters
    ----------
    noll_indices : int or sequence of int
        Noll's sequential index (see [1]_) of each Zernike mode.
    coefficients : float or sequence of float
        Coefficient in radians for each mode listed in `noll_indices`, i.e. the
        weight of the corresponding Noll-normalized (unit RMS over the unit disk)
        Zernike polynomial in the phase sum. Must have the same number of elements
        as `noll_indices`.
    mesh_size : int
        The size of the mesh grid for the pupil field.

    Returns
    -------
    NDArray[Float]
        The Zernike phase evaluated on the normalized pupil mesh.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices

    """
    if zernike_radial_cpu is None or fourier is None:
        raise ZernipaxNotInstalledError()

    noll_indices_arr = np.atleast_1d(np.asarray(noll_indices, dtype=int))
    coefficients_arr = np.atleast_1d(np.asarray(coefficients, dtype=float))
    if noll_indices_arr.shape != coefficients_arr.shape:
        raise ValueError(
            "noll_indices and coefficients must have the same number of elements, "
            f"got {noll_indices_arr.size} and {coefficients_arr.size}."
        )

    nm = np.array([_noll_to_nm(int(j)) for j in noll_indices_arr])
    n_vals, m_vals = nm[:, 0], nm[:, 1]

    rho, theta = zernike_pupil_coordinates(mesh_size)
    rho_flat = rho.ravel()
    theta_flat = theta.ravel()

    radial = np.asarray(zernike_radial_cpu(rho_flat, n_vals, m_vals))
    angular = np.asarray(fourier(theta_flat[:, np.newaxis], m_vals))

    # Noll normalization: unit RMS over the unit disk.
    normalization = np.where(m_vals == 0, np.sqrt(n_vals + 1), np.sqrt(2 * (n_vals + 1)))

    phase_flat = (radial * angular * normalization) @ coefficients_arr
    return phase_flat.reshape(mesh_size, mesh_size).astype(Float)
