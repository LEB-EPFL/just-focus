import math

import numpy as np
import pytest

from leb.just_focus import HalfmoonPhase, InputField, Polarization, float_dtype, gaussian_amplitude, phase_ramp


def test_gaussian_amplitude_shape_and_dtype():
    mesh_size = 8
    amplitude_x, amplitude_y = gaussian_amplitude((0.0, 0.0), 1.0, mesh_size)

    assert amplitude_x.shape == (mesh_size, mesh_size)
    assert amplitude_y.shape == (mesh_size, mesh_size)


def test_phase_ramp_shape_and_dtype():
    mesh_size = 8
    ramp = phase_ramp((0.5, -0.25), mesh_size)

    assert ramp.shape == (mesh_size, mesh_size)
    assert ramp.dtype == float_dtype()


def test_phase_ramp_zero_at_center():
    mesh_size = 65  # odd so that the linspace includes an exact 0.0 sample
    ramp = phase_ramp((1.0, 1.0), mesh_size)

    center = mesh_size // 2
    assert ramp[center, center] == pytest.approx(0.0)


def test_phase_ramp_value_at_pupil_edges():
    mesh_size = 65
    tilt_x, tilt_y = 0.7, -1.3
    ramp = phase_ramp((tilt_x, tilt_y), mesh_size)

    center = mesh_size // 2
    # np.meshgrid default "xy" indexing: px varies along columns, py along rows.
    assert ramp[center, -1] == pytest.approx(tilt_x)   # px = +1, py = 0
    assert ramp[center, 0] == pytest.approx(-tilt_x)   # px = -1, py = 0
    assert ramp[-1, center] == pytest.approx(tilt_y)   # py = +1, px = 0
    assert ramp[0, center] == pytest.approx(-tilt_y)   # py = -1, px = 0


def test_with_phase_ramp_adds_to_existing_phase_without_mutating_original():
    mesh_size = 65
    original = InputField.gaussian_halfmoon_pupil(
        beam_center_pupil=(0.0, 0.0),
        waist_pupil=1.0,
        mesh_size=mesh_size,
        polarization=Polarization.LINEAR_Y,
        orientation=HalfmoonPhase.HORIZONTAL,
        phase=np.pi,
    )
    original_phase_x = original.phase_x.copy()
    original_phase_y = original.phase_y.copy()

    tilt_pupil = (0.5, -0.25)
    tilted = original.with_phase_ramp(tilt_pupil)
    ramp = phase_ramp(tilt_pupil, mesh_size)

    assert np.allclose(tilted.phase_x, original_phase_x + ramp)
    assert np.allclose(tilted.phase_y, original_phase_y + ramp)
    # The original InputField must be untouched.
    assert np.array_equal(original.phase_x, original_phase_x)
    assert np.array_equal(original.phase_y, original_phase_y)
    # Amplitude and polarization pass through unchanged.
    assert np.array_equal(tilted.amplitude_x, original.amplitude_x)
    assert np.array_equal(tilted.amplitude_y, original.amplitude_y)
    assert np.array_equal(tilted.polarization_x, original.polarization_x)
    assert np.array_equal(tilted.polarization_y, original.polarization_y)


def test_with_phase_ramp_dtype_preserved():
    mesh_size = 32
    field = InputField.uniform_pupil(mesh_size, Polarization.LINEAR_X).with_phase_ramp((1.0, 1.0))

    assert field.phase_x.dtype == float_dtype()
    assert field.phase_y.dtype == float_dtype()


def test_polarization_plus_45_arrays():
    mesh_size = 4
    polarization_x, polarization_y = Polarization.LINEAR_PLUS_45.arrays(mesh_size)

    expected = 1 / math.sqrt(2)
    assert np.allclose(polarization_x, expected)
    assert np.allclose(polarization_y, expected)


def test_polarization_minus_45_arrays():
    mesh_size = 4
    polarization_x, polarization_y = Polarization.LINEAR_MINUS_45.arrays(mesh_size)

    expected = 1 / math.sqrt(2)
    assert np.allclose(polarization_x, expected)
    assert np.allclose(polarization_y, -expected)
