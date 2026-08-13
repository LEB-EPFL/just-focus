import importlib.util

import numpy as np
import pytest

from leb.just_focus import InputField, Polarization, backend
from leb.just_focus.zernike import ZernipaxNotInstalledError, _noll_to_nm, _zernike_basis

HAS_ZERNIPAX = importlib.util.find_spec("zernipax") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch

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
def test_with_zernike_modes_raises_without_zernipax():
    field = InputField.uniform_pupil(8, Polarization.LINEAR_X)
    with pytest.raises(ZernipaxNotInstalledError):
        field.with_zernike_modes(noll_indices=4, coefficients=1.0)


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_with_zernike_modes_shape_and_dtype():
    mesh_size = 16
    field = InputField.uniform_pupil(mesh_size, Polarization.LINEAR_X)
    aberrated = field.with_zernike_modes(noll_indices=[4, 11], coefficients=[0.5, -0.25])

    assert aberrated.phase_x.shape == (mesh_size, mesh_size)
    assert aberrated.phase_x.dtype == np.float64


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_zernike_basis_defocus_value_at_center():
    # Noll j=4 is defocus: n=2, m=0, R_2^0(rho) = 2*rho**2 - 1, Noll-normalized by
    # sqrt(n + 1) = sqrt(3). At the pupil center (rho=0) this evaluates to -sqrt(3).
    mesh_size = 65  # odd so the mesh includes the exact center (rho=0)
    basis = _zernike_basis(4, mesh_size)

    center = mesh_size // 2
    center_flat_index = center * mesh_size + center
    assert basis[center_flat_index, 0] == pytest.approx(-np.sqrt(3))


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_with_zernike_modes_adds_to_existing_phase_without_mutating_original():
    mesh_size = 16
    original = InputField.uniform_pupil(mesh_size, Polarization.LINEAR_Y)
    original_phase_x = original.phase_x.copy()
    original_phase_y = original.phase_y.copy()

    coefficients = np.array([0.3, -0.1])
    aberrated = original.with_zernike_modes(noll_indices=[4, 6], coefficients=coefficients)
    expected = (_zernike_basis([4, 6], mesh_size) @ coefficients).reshape(mesh_size, mesh_size)

    assert np.allclose(aberrated.phase_x, original_phase_x + expected)
    assert np.allclose(aberrated.phase_y, original_phase_y + expected)
    # The original InputField must be untouched.
    assert np.array_equal(original.phase_x, original_phase_x)
    assert np.array_equal(original.phase_y, original_phase_y)


@pytest.mark.skipif(not HAS_ZERNIPAX, reason="requires the optional zernipax dependency")
def test_zernike_basis_returns_independent_copies():
    # Each call must be independently mutable and unaffected by the internal
    # cache: mutating one result must not change the value of a later call.
    first = _zernike_basis([4, 11], 16)
    first[:] = 0.0
    second = _zernike_basis([4, 11], 16)

    assert not np.allclose(second, 0.0)


@pytest.mark.skipif(
    not (HAS_ZERNIPAX and HAS_TORCH), reason="requires the optional zernipax and torch dependencies"
)
def test_with_zernike_modes_coefficients_carry_torch_autograd():
    # This is the behavior that makes Zernike coefficients usable as Pyro latent
    # variables: gradients must flow from a downstream loss back into a
    # `coefficients` tensor, even though the basis itself is still built via
    # zernipax/JAX (which carries no autograd history).
    backend.set_backend("torch", precision="float64")

    mesh_size = 16
    field = InputField.uniform_pupil(mesh_size, Polarization.LINEAR_X)
    coefficients = torch.tensor([0.5, -0.2], dtype=torch.float64, requires_grad=True)

    aberrated = field.with_zernike_modes(noll_indices=[4, 11], coefficients=coefficients)

    assert isinstance(aberrated.phase_x, torch.Tensor)
    assert aberrated.phase_x.requires_grad

    aberrated.phase_x.sum().backward()

    assert coefficients.grad is not None
    assert torch.all(torch.isfinite(coefficients.grad))
    # The gradient of sum(basis @ c) w.r.t. c is the column-sum of the basis.
    basis = torch.as_tensor(_zernike_basis([4, 11], mesh_size), dtype=torch.float64)
    assert torch.allclose(coefficients.grad, basis.sum(dim=0))
