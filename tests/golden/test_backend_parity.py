"""Regression tests comparing each backend's output against frozen golden data.

The golden `.npz` files under `data/` were generated once from the
pre-refactor, NumPy-only version of this package (see
`generate_golden_data.py`) and are the external oracle both backends are
checked against here: the NumPy path must reproduce them essentially
exactly (any drift signals a refactor bug, not a numerical-method
difference), while the PyTorch path is allowed a looser, still-tight
tolerance (different FFT/math library, some ULP-level divergence expected).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cases import case_names, build_case, PADDING_FACTOR
from leb.just_focus import set_backend
from leb.just_focus.backend import be

HAS_TORCH = importlib.util.find_spec("torch") is not None

DATA_DIR = Path(__file__).parent / "data"

NUMPY_TOLERANCE = {"atol": 1e-10, "rtol": 1e-8}
TORCH_TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}


def _load_golden(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA_DIR / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def _run_case(name: str) -> dict[str, np.ndarray]:
    case = build_case(name)
    result = case.pupil.propgate(case.z_um, case.inputs, padding_factor=PADDING_FACTOR)
    return {
        "field_x": be.to_numpy(result.field_x),
        "field_y": be.to_numpy(result.field_y),
        "field_z": be.to_numpy(result.field_z),
        "x_um": be.to_numpy(result.x_um),
        "y_um": be.to_numpy(result.y_um),
        "intensity": be.to_numpy(result.intensity(normalize=False)),
    }


@pytest.mark.parametrize("case_name", case_names())
def test_numpy_matches_golden(case_name: str) -> None:
    set_backend("numpy")
    golden = _load_golden(case_name)
    actual = _run_case(case_name)
    for key, expected in golden.items():
        np.testing.assert_allclose(
            actual[key], expected, **NUMPY_TOLERANCE, err_msg=f"{case_name}: {key} mismatch"
        )


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
@pytest.mark.parametrize("case_name", case_names())
def test_torch_matches_golden(case_name: str) -> None:
    set_backend("torch")
    golden = _load_golden(case_name)
    actual = _run_case(case_name)
    for key, expected in golden.items():
        np.testing.assert_allclose(
            actual[key], expected, **TORCH_TOLERANCE, err_msg=f"{case_name}: {key} mismatch"
        )
