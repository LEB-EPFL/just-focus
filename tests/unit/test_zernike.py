import importlib.util

import numpy as np
import pytest

from leb.just_focus import InputField, Polarization
from leb.just_focus.zernike import ZernipaxNotInstalledError, _noll_to_nm, zernike_phase

HAS_ZERNIPAX = importlib.util.find_spec("zernipax") is not None

# Noll's original j <-> (n, m) table, j = 1..15.
# https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices
# equivalently, OEIS A176988: https://oeis.org/A176988
NOLL_TABLE = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, -2),
    6: (2, 2),
    7: (3, -1),
    8: (3, 1),
    9: (3, -3),
    10: (3, 3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
}


@pytest.mark.parametrize("j, nm", NOLL_TABLE.items())
def test_noll_to_nm_matches_known_table(j, nm):
    assert _noll_to_nm(j) == nm


def test_noll_to_nm_produces_valid_indices():
    # (n, m) must be a valid Zernike index pair: |m| <= n and (n - m) even.
    for j in range(1, 200):
        n, m = _noll_to_nm(j)
        assert abs(m) <= n
        assert (n - m) % 2 == 0


def test_noll_to_nm_is_a_bijection():
    # Distinct Noll indices must map to distinct (n, m) pairs.
    pairs = [_noll_to_nm(j) for j in range(1, 200)]
    assert len(set(pairs)) == len(pairs)


def test_noll_to_nm_invalid_index_raises():
    with pytest.raises(ValueError):
        _noll_to_nm(0)
    with pytest.raises(ValueError):
        _noll_to_nm(-3)


@pytest.mark.skipif(HAS_ZERNIPAX, reason="exercises the missing-dependency error path")
def test_zernike_phase_raises_without_zernipax():
    with pytest.raises(ZernipaxNotInstalledError):
        zernike_phase(noll_indices=4, coefficients=1.0, mesh_size=8)


@pytest.mark.skipif(HAS_ZERNIPAX, reason="exercises the missing-dependency error path")
def test_with_zernike_modes_raises_without_zernipax():
    field = InputField.uniform_pupil(8, Polarization.LINEAR_X)
    with pytest.raises(ZernipaxNotInstalledError):
        field.with_zernike_modes(noll_indices=4, coefficients=1.0)


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_zernike_phase_shape_and_dtype():
    mesh_size = 16
    phase = zernike_phase(noll_indices=[4, 11], coefficients=[0.5, -0.25], mesh_size=mesh_size)

    assert phase.shape == (mesh_size, mesh_size)
    assert phase.dtype == np.float64


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_zernike_phase_defocus_value_at_center():
    # Noll j=4 is defocus: n=2, m=0, R_2^0(rho) = 2*rho**2 - 1, Noll-normalized by
    # sqrt(n + 1) = sqrt(3). At the pupil center (rho=0) this evaluates to -sqrt(3).
    mesh_size = 65  # odd so the mesh includes the exact center (rho=0)
    coefficient = 0.7
    phase = zernike_phase(noll_indices=4, coefficients=coefficient, mesh_size=mesh_size)

    center = mesh_size // 2
    assert phase[center, center] == pytest.approx(-coefficient * np.sqrt(3))


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_with_zernike_modes_adds_to_existing_phase_without_mutating_original():
    mesh_size = 16
    original = InputField.uniform_pupil(mesh_size, Polarization.LINEAR_Y)
    original_phase_x = original.phase_x.copy()
    original_phase_y = original.phase_y.copy()

    aberrated = original.with_zernike_modes(noll_indices=[4, 6], coefficients=[0.3, -0.1])
    expected = zernike_phase([4, 6], [0.3, -0.1], mesh_size)

    assert np.allclose(aberrated.phase_x, original_phase_x + expected)
    assert np.allclose(aberrated.phase_y, original_phase_y + expected)
    # The original InputField must be untouched.
    assert np.array_equal(original.phase_x, original_phase_x)
    assert np.array_equal(original.phase_y, original_phase_y)
