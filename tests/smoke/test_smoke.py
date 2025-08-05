import numpy as np

from leb.just_focus import InputField, Polarization, Pupil
from leb.just_focus.plots import plot_inputs


def test_smoke() -> None:
    """Smoke test to ensure everything runs without errors."""
    np.seterr("raise")

    mesh_size = 32
    pupil = Pupil(
        na=1.4,
        refractive_index=1.518,
        wavelength_um=0.561,
        mesh_size=mesh_size
    )
    inputs = InputField.uniform_pupil(mesh_size, Polarization.CIRCULAR_LEFT)

    results = pupil.propgate(0.0, inputs, padding_factor=5)

    results.intensity()

    plot_inputs(inputs, pupil, show=False)
