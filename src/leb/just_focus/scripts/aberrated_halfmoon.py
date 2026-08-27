"""Generate a half-moon pupil with Zernike polynomial phase aberrations and visualize the
results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from leb.just_focus import HalfmoonPhase, InputField, Polarization, Pupil, Stop
from leb.just_focus.zernike import zernike_pupil_coordinates


def _add_stop_outline(ax, radius_mm: float) -> None:
    ax.add_artist(Circle((0, 0), radius=radius_mm, color="k", fill=False, linewidth=2))


def main(plot=True) -> None:
    mesh_size = 256
    pupil = Pupil(
        na=1.4,
        refractive_index=1.518,
        wavelength_um=0.561,
        mesh_size=mesh_size,
        stop=Stop.TANH,
        stop_radius_pupil=1.0,
    )

    halfmoon = InputField.gaussian_halfmoon_pupil(
        beam_center_pupil=(0.0, 0.0),
        waist_pupil=2.0,
        mesh_size=mesh_size,
        polarization=Polarization.LINEAR_Y,
        orientation=HalfmoonPhase.MINUS_45,
        phase=np.pi,
        phase_mask_center=(0.0, 0.0),
    )

    # Random low-order aberration: all Noll modes 1-21, each weighted randomly in radians.
    inputs = halfmoon.with_zernike_modes(
        noll_indices=list(range(1, 22)),
        coefficients=np.random.uniform(-0.1, 0.1, size=21).tolist(),
    )

    results = pupil.propagate(0.0, inputs, padding_factor=4)

    # Zero out input values outside the pupil aperture before plotting; Zernike
    # polynomials are only defined on the unit disk.
    rho, _ = zernike_pupil_coordinates(mesh_size)
    pupil_mask = rho <= pupil.stop_radius_pupil
    amplitude_x = np.where(pupil_mask, inputs.amplitude_x, 0.0)
    amplitude_y = np.where(pupil_mask, inputs.amplitude_y, 0.0)
    phase_x = np.where(pupil_mask, inputs.phase_x, 0.0)
    phase_y = np.where(pupil_mask, inputs.phase_y, 0.0)
    polarization_x = np.where(pupil_mask, inputs.polarization_x, 0.0)
    polarization_y = np.where(pupil_mask, inputs.polarization_y, 0.0)

    _, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs[0, 0].imshow(
        amplitude_x,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    _add_stop_outline(axs[0, 0], pupil.stop_radius_mm)
    axs[0, 0].set_ylabel("y, mm")
    axs[0, 0].set_title("Amplitude, x")

    axs[0, 1].imshow(
        amplitude_y,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    _add_stop_outline(axs[0, 1], pupil.stop_radius_mm)
    axs[0, 1].set_title("Amplitude, y")

    axs[0, 2].imshow(
        phase_x,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    _add_stop_outline(axs[0, 2], pupil.stop_radius_mm)
    axs[0, 2].set_title("Phase, x (aberrated)")

    axs[0, 3].imshow(
        phase_y,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    _add_stop_outline(axs[0, 3], pupil.stop_radius_mm)
    axs[0, 3].set_title("Phase, y (aberrated)")

    axs[1, 0].imshow(
        np.abs(polarization_x),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    _add_stop_outline(axs[1, 0], pupil.stop_radius_mm)
    axs[1, 0].set_title("Polarization, x")
    axs[1, 0].set_xlabel("x, mm")
    axs[1, 0].set_ylabel("y, mm")

    axs[1, 1].imshow(
        np.abs(polarization_y),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    _add_stop_outline(axs[1, 1], pupil.stop_radius_mm)
    axs[1, 1].set_title("Polarization, y")
    axs[1, 1].set_xlabel("x, mm")

    axs[1, 2].imshow(pupil.stop_arr, vmin=0, vmax=1)
    axs[1, 2].set_title("Stop")
    axs[1, 2].set_xlabel("x, mm")

    axs[1, 3].imshow(
        results.intensity(normalize=True),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(results.x_um[0], results.x_um[-1], results.y_um[0], results.y_um[-1])
    )
    axs[1, 3].set_title("Intensity")
    axs[1, 3].set_xlabel(r"x, $\mu m$")
    axs[1, 3].set_xlim(-1, 1)
    axs[1, 3].set_ylim(-1, 1)

    if plot:
        plt.show()


if __name__ == "__main__":
    np.seterr("raise")
    main()
