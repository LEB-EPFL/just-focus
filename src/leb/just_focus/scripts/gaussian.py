"""Generate the focal fields from a Gaussian beam in a pupil and visualize the results."""

import matplotlib.pyplot as plt
import numpy as np

from leb.just_focus import InputField, Polarization, Pupil, Stop


def main(plot=True) -> None:
    mesh_size = 64
    pupil = Pupil(
        na=1.4,
        refractive_index=1.518,
        wavelength_um=0.561,
        mesh_size=mesh_size,
        stop=Stop.TANH,
    )

    inputs = InputField.gaussian_pupil(
        beam_center=(0.0, 0.0),
        waist=5.0,
        mesh_size=mesh_size,
        polarization=Polarization.LINEAR_X,
    )

    results = pupil.propgate(0.0, inputs, padding_factor=4)

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(
        np.abs(results.field_x),
        origin="lower",
        extent=(results.x_um[0], results.x_um[-1], results.y_um[0], results.y_um[-1]),
    )
    axs[1].imshow(
        np.angle(results.field_x),
        origin="lower",
        extent=(results.x_um[0], results.x_um[-1], results.y_um[0], results.y_um[-1]),
    )
    
    if plot:
        plt.show()


if __name__ == "__main__":
    np.seterr("raise")
    main()
