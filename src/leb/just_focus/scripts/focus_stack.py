"""Generate an axial focus stack from a linearly polarized Gaussian beam and visualize
the intensity in the xy, xz, and yz planes."""

import matplotlib.pyplot as plt
import numpy as np

from leb.just_focus import InputField, Polarization, Pupil, Stop


def main(plot=True) -> None:
    mesh_size = 64
    padding_factor = 4
    z_um = np.linspace(-1.0, 1.0, 81)

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
        polarization=Polarization.LINEAR_X,
    )

    x_um = y_um = None
    xy_intensity = None
    xz_intensity = []
    yz_intensity = []
    z0_index = int(np.argmin(np.abs(z_um)))
    for i, z in enumerate(z_um):
        results = pupil.propagate(float(z), inputs, padding_factor=padding_factor)
        intensity = results.intensity(normalize=False)

        if x_um is None:
            x_um, y_um = results.x_um, results.y_um
        if i == z0_index:
            xy_intensity = intensity

        # Focal field mesh samples the origin, so the center pixel is the on-axis
        # (x=0, y=0) sample.
        center = intensity.shape[0] // 2
        xz_intensity.append(intensity[center, :])
        yz_intensity.append(intensity[:, center])

    xz_intensity = np.array(xz_intensity)  # shape (len(z_um), len(x_um))
    yz_intensity = np.array(yz_intensity)  # shape (len(z_um), len(y_um))

    # Normalize all three slices together so relative intensity across z is preserved.
    vmax = max(xy_intensity.max(), xz_intensity.max(), yz_intensity.max())

    _, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(
        xy_intensity / vmax,
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        extent=(x_um[0], x_um[-1], y_um[0], y_um[-1]),
    )
    axs[0].set_title(f"xy, z = {z_um[z0_index]:.2f} " r"$\mu m$")
    axs[0].set_xlabel(r"x, $\mu m$")
    axs[0].set_ylabel(r"y, $\mu m$")
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)

    # xz_intensity/yz_intensity already have z varying along axis 0 (rows), so no
    # transpose is needed to put z on the vertical axis.
    axs[1].imshow(
        xz_intensity / vmax,
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        extent=(x_um[0], x_um[-1], z_um[0], z_um[-1]),
    )
    axs[1].set_title("xz, y = 0")
    axs[1].set_xlabel(r"x, $\mu m$")
    axs[1].set_ylabel(r"z, $\mu m$")
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(z_um[0], z_um[-1])

    axs[2].imshow(
        yz_intensity / vmax,
        vmin=0,
        vmax=1,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        extent=(y_um[0], y_um[-1], z_um[0], z_um[-1]),
    )
    axs[2].set_title("yz, x = 0")
    axs[2].set_xlabel(r"y, $\mu m$")
    axs[2].set_ylabel(r"z, $\mu m$")
    axs[2].set_xlim(-1, 1)
    axs[2].set_ylim(z_um[0], z_um[-1])

    plt.tight_layout()

    if plot:
        plt.show()


if __name__ == "__main__":
    np.seterr("raise")
    main()
