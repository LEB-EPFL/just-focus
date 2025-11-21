"""Input fields for the propagation algorithm."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .dtypes import Complex, Float

def get_normed_coords(mesh_size: int, center: tuple[float, float] = (0.0, 0.0)) -> tuple[NDArray[Float], NDArray[Float]]:
    normed_coords = np.linspace(-1, 1, mesh_size)
    x, y = np.meshgrid(normed_coords, normed_coords)
    x0: float = center[0]
    y0: float = center[1]
    x -= x0
    y -= y0
    return x, y

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
        phase_mask_center: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[NDArray[Float], NDArray[Float]]:
        
        x, y = get_normed_coords(mesh_size, phase_mask_center)

        phase_x = np.zeros((mesh_size, mesh_size), dtype=Float)
        match self:
            case HalfmoonPhase.HORIZONTAL:
                mask = x >= 0
            case HalfmoonPhase.VERTICAL:
                mask = y >= 0
            case HalfmoonPhase.PLUS_45:
                mask = (x + y) >= 0
            case HalfmoonPhase.MINUS_45:
                mask = (x - y) >= 0 
            
        phase_x[mask] = phase
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

        x, y = get_normed_coords(mesh_size, beam_center)
        amplitude_x = np.exp(-x**2 / waist_x**2 - y**2 / waist_y**2)
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
        phase_jump_shift: tuple[float, float] = (0.0, 0.0),
        phase_mask_center: tuple[float, float] = (0.0, 0.0),
        phase_mask_size: float | tuple[float, float] = (1.0, 1.0),
        grating_period: float = None,
        grating_angle: float = 0.0,
        noise_level: float = 0.0
    ) -> InputField:
        """Create a Gaussian pupil field with a half-moon phase mask.
        
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
        orientation : HalfmoonPhase
            The orientation of the half-moon phase mask.
        phase : float
            The phase shift applied by the half-moon mask, in radians.
        phase_jump_shift : tuple of float
            The shift of the phase jump center relative to the phase mask center, 
            in normalized pupil coordinates (x, y).
        phase_mask_center : tuple of float
            The center of the phase mask in normalized pupil coordinates (x, y).
        phase_mask_size : float or tuple of float
            Size of the phase mask in normalized pupil coordinates. If a single float
            is provided, it is used for both x and y dimensions.
        grating_period : float
            Period of the blazed grating in normalized pupil coordinates. If None or
            non-positive, no grating is applied.
        grating_angle : float
            Angle of the blazed grating, in radians.
        noise_level : float
            Standard deviation of the Gaussian noise to be added to the phase arrays.
            It is expressed as a fraction of 2π.
        
        Returns
        -------
        InputField
            The input field with Gaussian amplitude, half-moon phase mask, and
            specified polarization.

        """
        polarization_x, polarization_y = polarization.arrays(mesh_size)
        amplitude_x, amplitude_y = cls._gaussian_amplitude(beam_center, waist, mesh_size)

        phase_jump_center = (phase_mask_center[0] + phase_jump_shift[0],
                             phase_mask_center[1] + phase_jump_shift[1])
        phase_x, phase_y = orientation.arrays(mesh_size, phase, phase_jump_center)
        phase_x, phase_y = cls.add_noise(phase_x, phase_y, noise_level)
        phase_x, phase_y = cls.add_blazed_grating(phase_x, phase_y, grating_period, grating_angle, phase_mask_center)
        phase_x, phase_y = cls.change_mask_size(phase_x, phase_y, phase_mask_size, phase_mask_center)

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
    
    @staticmethod
    def add_noise(phase_x, phase_y, noise_level: float = 0.0) -> tuple[NDArray[float], NDArray[float]]:
        """Add random noise to the phase arrays.
        
        Parameters
        ----------
        phase_x : NDArray[float]
            Phase array for the x-direction.
        phase_y : NDArray[float]
            Phase array for the y-direction.
        noise_level : float
            Standard deviation of the Gaussian noise to be added to the phase arrays.
            It is expressed as a fraction of 2π.
        
        Returns
        -------
        tuple[NDArray[float], NDArray[float]]
            Modified phase arrays with added noise.
        """
        noise_x = np.random.normal(0, noise_level * 2 * np.pi, phase_x.shape)
        noise_y = np.random.normal(0, noise_level * 2 * np.pi, phase_y.shape)

        mask_x = phase_x > 0
        mask_y = phase_y > 0

        phase_x[mask_x] += noise_x[mask_x]
        phase_y[mask_y] += noise_y[mask_y]

        phase_x = np.mod(phase_x, 2 * np.pi)
        phase_y = np.mod(phase_y, 2 * np.pi)

        return phase_x, phase_y

        

    @staticmethod
    def change_mask_size(phase_x: NDArray[float], 
                         phase_y: NDArray[float],
                         phase_mask_size: float | tuple[float, float],
                         phase_mask_center: tuple[float, float]
                        ) -> tuple[NDArray[float], NDArray[float]]:
        """Change the size of the half-moon phase mask applied to the phase arrays.
        
        Parameters
        ----------
        phase_x : NDArray[float]
            Phase array for the x-direction.
        phase_y : NDArray[float]
            Phase array for the y-direction.
        phase_mask_size : float or tuple of float
            Size of the phase mask in normalized pupil coordinates. If a single
            float is provided, it is used for both x and y dimensions.
        phase_mask_center : tuple of float
            Center of the phase mask in normalized pupil coordinates (x, y).
        
        Returns
        -------
        tuple[NDArray[float], NDArray[float]]
            Modified phase arrays with the adjusted phase mask size.
        """
        if isinstance(phase_mask_size, (int, float)):
            phase_mask_size_x = phase_mask_size_y = phase_mask_size
        else:
            phase_mask_size_x, phase_mask_size_y = phase_mask_size

        mesh_size = phase_x.shape[0]
        x, y = get_normed_coords(mesh_size, phase_mask_center)

        mask = x**2 <= phase_mask_size_x**2
        mask *= y**2 <= phase_mask_size_y**2
        phase_x *= mask
        phase_y *= mask

        return phase_x, phase_y
    
    @staticmethod
    def add_blazed_grating(phase_x: NDArray[float],
                           phase_y: NDArray[float],
                           grating_period: float | None,
                           grating_angle: float,
                           phase_mask_center: tuple[float, float]=(0.0, 0.0)
                          ) -> tuple[NDArray[float], NDArray[float]]:
        """Add a blazed grating to the phase arrays.
        
        Parameters
        ----------
        phase_x : NDArray[float]
            Phase array for the x-direction.
        phase_y : NDArray[float]
            Phase array for the y-direction.
        grating_period : float
            Period of the blazed grating in normalized pupil coordinates.
        grating_angle : float
            Angle of the blazed grating in radians.
        phase_mask_center : tuple of float
            Center of the phase mask in normalized pupil coordinates (x, y).
        
        Returns
        -------
        tuple[NDArray[float], NDArray[float]]
            Modified phase arrays with the blazed grating added.
        """
        if grating_period is not None and grating_period > 0.0:
            mesh_size = phase_x.shape[0]
            x, y = get_normed_coords(mesh_size, phase_mask_center)

            kx = (2 * np.pi / grating_period) * np.cos(grating_angle)
            ky = (2 * np.pi / grating_period) * np.sin(grating_angle)

            blazed_grating = np.mod(kx * x + ky * y, 2 * np.pi)

            phase_x = np.mod(phase_x, 2 * np.pi)
            phase_y = np.mod(phase_y, 2 * np.pi)
            
            phase_x += blazed_grating
            phase_y += blazed_grating

            phase_x = np.mod(phase_x, 2 * np.pi)
            phase_y = np.mod(phase_y, 2 * np.pi)

            return phase_x, phase_y
        else:
            return phase_x, phase_y