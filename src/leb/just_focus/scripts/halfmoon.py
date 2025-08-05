"""Generate a half-moon pupil with an off center Gaussian beam and visualize the results."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from leb.just_focus import HalfmoonPhase, InputField, Polarization, Pupil, Stop


def main(plot=True) -> None:
    mesh_size = 64
    pupil = Pupil(
        na=1.4,
        refractive_index=1.518,
        wavelength_um=0.561,
        mesh_size=mesh_size,
        stop=Stop.TANH,
    )

    inputs = InputField.gaussian_halfmoon_pupil(
        beam_center=(0.0, 0.5),
        waist=2.0,
        mesh_size=mesh_size,
        polarization=Polarization.LINEAR_Y,
        orientation=HalfmoonPhase.MINUS_45,
        phase=np.pi,
        phase_mask_center=(0.0, 0.0),
    )

    results = pupil.propgate(0.0, inputs, padding_factor=4)

    _, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs[0, 0].imshow(
        inputs.amplitude_x,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[0, 0].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[0, 0].set_ylabel("y, mm")
    axs[0, 0].set_title("Amplitude, x")

    axs[0, 1].imshow(
        inputs.amplitude_y,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[0, 1].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[0, 1].set_title("Amplitude, y")

    axs[0, 2].imshow(
        inputs.phase_x,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[0, 2].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[0, 2].set_title("Phase, x")

    axs[0, 3].imshow(
        inputs.phase_y,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[0, 3].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[0, 3].set_title("Phase, y")

    axs[1, 0].imshow(
        np.abs(inputs.polarization_x),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[1, 0].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
    axs[1, 0].set_title("Polarization, x")
    axs[1, 0].set_xlabel("x, mm")
    axs[1, 0].set_ylabel("y, mm")

    axs[1, 1].imshow(
        np.abs(inputs.polarization_y),
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(pupil.x_mm[0], pupil.x_mm[-1], pupil.y_mm[0], pupil.y_mm[-1]),
    )
    axs[1, 1].add_artist(Circle((0, 0), radius=pupil.stop_radius_mm, color='k', fill=False, linewidth=2))
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
