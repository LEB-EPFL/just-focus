"""Input fields for the propagation algorithm."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .dtypes import Complex, Float


class Polarization(StrEnum):
    LINEAR_X = "linear_x"
    LINEAR_Y = "linear_y"
    CIRCULAR_LEFT = "circular_left"
    CIRCULAR_RIGHT = "circular_right"

    def arrays(self, mesh_size: int) -> tuple[NDArray[Complex], NDArray[Complex]]:
        match self:
            case Polarization.LINEAR_X:
                polarization_x = np.ones((mesh_size, mesh_size), dtype=Complex)
                polarization_y = np.zeros((mesh_size, mesh_size), dtype=Complex)
            case Polarization.LINEAR_Y:
                polarization_x = np.zeros((mesh_size, mesh_size), dtype=Complex)
                polarization_y = np.ones((mesh_size, mesh_size), dtype=Complex)
            case Polarization.CIRCULAR_LEFT:
                polarization_x = np.ones((mesh_size, mesh_size), dtype=Complex) / np.sqrt(2)
                polarization_y = 1j * np.ones((mesh_size, mesh_size), dtype=Complex) / np.sqrt(2)
            case Polarization.CIRCULAR_RIGHT:
                polarization_x = np.ones((mesh_size, mesh_size), dtype=Complex) / np.sqrt(2)
                polarization_y = -1j * np.ones((mesh_size, mesh_size), dtype=Complex) / np.sqrt(2)
            case _:
                raise ValueError(f"Unsupported polarization: {polarization}")
        
        return polarization_x, polarization_y


class HalfmoonPhase(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    PLUS_45 = "plus_45"
    MINUS_45 = "minus_45"

    def arrays(
        self,
        mesh_size: int,
        phase: float = np.pi,
        phase_mask_center: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[NDArray[Float], NDArray[Float]]:
        normed_coords = np.linspace(-1, 1, mesh_size)
        x, y = np.meshgrid(normed_coords, normed_coords)
        x0, y0 = phase_mask_center
        x -= x0
        y -= y0

        phase_x = np.zeros((mesh_size, mesh_size), dtype=Float)
        match self:
            case HalfmoonPhase.HORIZONTAL:
                phase_x[x >= 0] = phase
            case HalfmoonPhase.VERTICAL:
                phase_x[y >= 0] = phase
            case HalfmoonPhase.PLUS_45:
                phase_x[(x + y) >= 0] = phase
            case HalfmoonPhase.MINUS_45:
                phase_x[(x - y) >= 0] = phase

        phase_y = phase_x.copy()

        return phase_x, phase_y



@dataclass
class InputField:
    """Factory class for creating input fields for the pupil.

    Each direction may be specified independently, which models separate beam shaping
    elements for the x- and y-directions. In many common cases, the amplitudes and
    phases will be the same in both x- and y-directions and only the polarization will
    differ.
    
    Attributes
    ----------
    amplitude_x : NDArray[Float]
        The amplitude of the field for the x-direction.
    amplitude_y : NDArray[Float]
        The amplitude of the field for the y-direction.
    phase_x : NDArray[Float]
        The phase of the field for the x-direction.
    phase_y : NDArray[Float]
        The phase of the field for the y-direction.
    polarization_x : NDArray[Complex]
        The polarization state of the field for the x-direction.
    polarization_y : NDArray[Complex]
        The polarization state of the field for the y-direction.

    Methods
    -------
    gaussian_pupil(beam_center, waist, mesh_size, polarization)
        Create a Gaussian pupil field with a specified waist size.
    uniform_pupil(mesh_size, polarization)
        Create a uniform pupil field with specified polarization.

    """
    amplitude_x: NDArray[Float]
    amplitude_y: NDArray[Float]
    phase_x: NDArray[Float]
    phase_y: NDArray[Float]
    polarization_x: NDArray[Complex]
    polarization_y: NDArray[Complex]

    @staticmethod
    def _gaussian_amplitude(
        beam_center: tuple[float, float],
        waist: float | tuple[float, float],
        mesh_size: int
    ) -> tuple[NDArray[Float], NDArray[Float]]:
        """Calculate a Gaussian amplitude for the pupil field."""
        if isinstance(waist, (int, float)):
            waist_x = waist_y = waist
        else:
            waist_x, waist_y = waist

        normed_coords = np.linspace(-1, 1, mesh_size)
        x, y = np.meshgrid(normed_coords, normed_coords)
        x0: float = beam_center[0]
        y0: float = beam_center[1]
        amplitude_x = np.exp(-(x - x0)**2 / waist_x**2 - (y - y0)**2 / waist_y**2)
        amplitude_y = np.copy(amplitude_x)

        return amplitude_x, amplitude_y

    @classmethod
    def gaussian_pupil(
        cls,
        beam_center: tuple[float, float],
        waist: float | tuple[float, float],
        mesh_size: int,
        polarization: Polarization
    ) -> InputField:
        """Create a Gaussian pupil field with a specified waist size.

        Parameters
        ----------
        beam_center : tuple of float
            The center of the Gaussian beam in normalized pupil coordinates (x, y).
        waist : float or tuple of float
            The waist size of the Gaussian beam in normalized pupil coordinates. If a
            single float is provided, it is used for both x and y dimensions.
        mesh_size : int
            The size of the mesh grid for the pupil field.
        polarization : Polarization
            The polarization state of the field.

        Returns
        -------
        InputField
            The input field with Gaussian amplitude and specified polarization.

        """
        polarization_x, polarization_y = polarization.arrays(mesh_size)
        amplitude_x, amplitude_y = cls._gaussian_amplitude(beam_center, waist, mesh_size)

        phase_x = np.zeros((mesh_size, mesh_size), dtype=Float)
        phase_y = np.zeros((mesh_size, mesh_size), dtype=Float)

        return InputField(
            amplitude_x=amplitude_x,
            amplitude_y=amplitude_y,
            phase_x=phase_x,
            phase_y=phase_y,
            polarization_x=polarization_x,
            polarization_y=polarization_y,
        )
    
    @classmethod
    def gaussian_halfmoon_pupil(
        cls,
        beam_center: tuple[float, float],
        waist: float | tuple[float, float],
        mesh_size: int,
        polarization:Polarization,
        orientation: HalfmoonPhase = HalfmoonPhase.HORIZONTAL,
        phase: float = np.pi,
        phase_mask_center: tuple[float, float] = (0.0, 0.0)
    ) -> InputField:
        polarization_x, polarization_y = polarization.arrays(mesh_size)
        amplitude_x, amplitude_y = cls._gaussian_amplitude(beam_center, waist, mesh_size)

        phase_x, phase_y = orientation.arrays(mesh_size, phase, phase_mask_center)

        return InputField(
            amplitude_x=amplitude_x,
            amplitude_y=amplitude_y,
            phase_x=phase_x,
            phase_y=phase_y,
            polarization_x=polarization_x,
            polarization_y=polarization_y,
        )

    @classmethod
    def uniform_pupil(cls, mesh_size: int, polarization: Polarization) -> InputField:
        polarization_x, polarization_y = polarization.arrays(mesh_size)
            
        amplitude_x = np.ones((mesh_size, mesh_size), dtype=Float)
        amplitude_y = np.ones((mesh_size, mesh_size), dtype=Float)
        phase_x = np.zeros((mesh_size, mesh_size), dtype=Float)
        phase_y = np.zeros((mesh_size, mesh_size), dtype=Float)

        return InputField(
            amplitude_x=amplitude_x,
            amplitude_y=amplitude_y,
            phase_x=phase_x,
            phase_y=phase_y,
            polarization_x=polarization_x,
            polarization_y=polarization_y,
        )
