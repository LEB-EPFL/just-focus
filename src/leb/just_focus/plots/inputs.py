"""Tools for plotting the inputs to a simulation.

Requires the "plot" extra to be installed.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from leb.just_focus import InputField, Pupil

def plot_inputs(inputs: InputField, pupil: Pupil, show: bool = True) -> None:
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

    axs[1, 3].remove()  # Remove the empty subplot

    if show:
        plt.show()
