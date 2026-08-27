"""Generate the focal fields from a Gaussian beam steered by a phase ramp and visualize the results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from leb.just_focus import InputField, Polarization, Pupil, Stop


def main(plot=True) -> None:
    mesh_size = 64
    tilt_pupil = (-2.0, 1.0)

    pupil = Pupil(
        na=1.4,
        refractive_index=1.518,
        wavelength_um=0.561,
        mesh_size=mesh_size,
        stop=Stop.TANH,
    )

    inputs = InputField.gaussian_pupil(
        beam_center_pupil=(0.0, 0.0),
        waist_pupil=1.0,
        mesh_size=mesh_size,
        polarization=Polarization.LINEAR_Y,
    ).with_phase_ramp(tilt_pupil)

    results = pupil.propagate(0.0, inputs, padding_factor=4)

    _, axs = plt.subplots(3, 2, figsize=(8, 10))
    plt.suptitle(f"tilt_pupil = {tilt_pupil}")

    # amplitude_x/y and phase_x/y are identical here (LINEAR_Y polarization,
    # ramp applied equally to both), so show one of each.
    axs[0, 0].imshow(
        inputs.amplitude_x,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[0, 0].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[0, 0].set_ylabel("y, mm")
    axs[0, 0].set_title("Amplitude")

    axs[0, 1].imshow(
        inputs.phase_x,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[0, 1].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[0, 1].set_title("Phase")

    axs[1, 0].imshow(
        np.abs(inputs.polarization_x),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[1, 0].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[1, 0].set_ylabel("y, mm")
    axs[1, 0].set_title("Polarization, x")

    axs[1, 1].imshow(
        np.abs(inputs.polarization_y),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[1, 1].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[1, 1].set_title("Polarization, y")

    axs[2, 0].imshow(pupil.stop_arr, vmin=0, vmax=1)
    axs[2, 0].set_title("Stop")
    axs[2, 0].set_xlabel("x, mm")
    axs[2, 0].set_ylabel("y, mm")

    axs[2, 1].imshow(
        results.intensity(normalize=True),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(results.x_um[0], results.x_um[-1], results.y_um[0], results.y_um[-1]),
    )
    axs[2, 1].set_title("Intensity")
    axs[2, 1].set_xlabel(r"x, $\mu m$")
    axs[2, 1].set_xlim(-1, 1)
    axs[2, 1].set_ylim(-1, 1)

    plt.tight_layout()

    if plot:
        plt.show()


if __name__ == "__main__":
    np.seterr("raise")
    main()
