import numpy as np
import pytest

from leb.just_focus import InputField, Polarization, Pupil


def test_pad_width():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    padding_factor = 2
    expected_padding = (3, 3)  # Add 3 elements to each side to get 2 * 2**2 = 8

    result = Pupil._pad_width(arr.shape, padding_factor=padding_factor)

    assert result[0] == expected_padding
    assert result[1] == expected_padding


def _make_pupil(mesh_size: int = 16) -> Pupil:
    return Pupil(na=1.4, refractive_index=1.518, wavelength_um=0.561, mesh_size=mesh_size)


def _on_axis_intensity(
    pupil: Pupil, z_um: float, inputs: InputField, padding_factor: int = 3
) -> float:
    field = pupil.propagate(z_um, inputs, padding_factor=padding_factor)
    intensity = field.intensity(normalize=False)
    center = intensity.shape[0] // 2, intensity.shape[1] // 2
    return intensity[center]


def test_propagate_intensity_changes_with_z():
    pupil = _make_pupil()
    inputs = InputField.uniform_pupil(pupil.mesh_size, Polarization.LINEAR_X)

    intensity_in_focus = pupil.propagate(0.0, inputs, padding_factor=3).intensity(normalize=False)
    intensity_defocused = pupil.propagate(0.3, inputs, padding_factor=3).intensity(normalize=False)

    assert not np.allclose(intensity_in_focus, intensity_defocused, atol=0)


def _symmetric_field_cases() -> list[tuple[str, InputField]]:
    mesh_size = 16
    return [
        ("uniform_linear_x", InputField.uniform_pupil(mesh_size, Polarization.LINEAR_X)),
        (
            "uniform_linear_plus_45",
            InputField.uniform_pupil(mesh_size, Polarization.LINEAR_PLUS_45),
        ),
        (
            "uniform_circular_left",
            InputField.uniform_pupil(mesh_size, Polarization.CIRCULAR_LEFT),
        ),
        (
            "gaussian_centered",
            InputField.gaussian_pupil((0.0, 0.0), 0.7, mesh_size, Polarization.LINEAR_X),
        ),
        (
            "gaussian_offcenter_circular",
            InputField.gaussian_pupil((0.3, 0.1), 0.7, mesh_size, Polarization.CIRCULAR_LEFT),
        ),
        (
            "halfmoon",
            InputField.gaussian_halfmoon_pupil((0.0, 0.0), 0.7, mesh_size, Polarization.LINEAR_X),
        ),
    ]


@pytest.mark.parametrize("case_name,inputs", _symmetric_field_cases())
@pytest.mark.parametrize("z_um", [0.1, 0.2, 0.4])
def test_propagate_on_axis_intensity_symmetric_about_focus(case_name, inputs, z_um):
    pupil = _make_pupil()

    intensity_pos_z = _on_axis_intensity(pupil, z_um, inputs)
    intensity_neg_z = _on_axis_intensity(pupil, -z_um, inputs)

    assert intensity_pos_z == pytest.approx(intensity_neg_z, rel=1e-3)


@pytest.mark.parametrize(
    "inputs",
    [
        InputField.uniform_pupil(16, Polarization.LINEAR_X),
        InputField.gaussian_pupil((0.0, 0.0), 0.7, 16, Polarization.LINEAR_X),
    ],
)
def test_propagate_on_axis_intensity_decreases_near_focus(inputs):
    pupil = _make_pupil()
    z_values_um = [0.0, 0.05, 0.1, 0.15, 0.2]

    intensities = [_on_axis_intensity(pupil, z_um, inputs) for z_um in z_values_um]

    assert all(intensities[i] > intensities[i + 1] for i in range(len(intensities) - 1))
