"""Input fields for the propagation algorithm."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .dtypes import Complex, Float
from .zernike import zernike_phase


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


def gaussian_amplitude(
    beam_center_pupil: tuple[float, float],
    waist_pupil: float | tuple[float, float],
    mesh_size: int
) -> tuple[NDArray[Float], NDArray[Float]]:
    """Compute a Gaussian amplitude for the pupil field.

    Parameters
    ----------
    beam_center_pupil : tuple of float
        The center of the Gaussian beam in normalized pupil coordinates (x, y).
    waist_pupil : float or tuple of float
        The waist size of the Gaussian beam in normalized pupil coordinates. If a
        single float is provided, it is used for both x and y dimensions.
    mesh_size : int
        The size of the mesh grid for the pupil field.

    Returns
    -------
    tuple of NDArray[Float]
        The Gaussian amplitude for the x- and y-directions, respectively.

    """
    if isinstance(waist_pupil, (int, float)):
        waist_x = waist_y = waist_pupil
    else:
        waist_x, waist_y = waist_pupil

    normed_coords = np.linspace(-1, 1, mesh_size)
    x, y = np.meshgrid(normed_coords, normed_coords)
    x0: float = beam_center_pupil[0]
    y0: float = beam_center_pupil[1]
    amplitude_x = np.exp(-(x - x0)**2 / waist_x**2 - (y - y0)**2 / waist_y**2)
    amplitude_y = np.copy(amplitude_x)

    return amplitude_x, amplitude_y


def phase_ramp(tilt_pupil: tuple[float, float], mesh_size: int) -> NDArray[Float]:
    """Compute a linear phase ramp (blazed grating) across the pupil.

    Parameters
    ----------
    tilt_pupil : tuple of float
        Phase tilt in radians at the pupil edge (px=1, py=1) along the pupil's
        x- and y-directions, i.e. (tilt_x, tilt_y).
    mesh_size : int
        The size of the mesh grid for the pupil field.

    Returns
    -------
    NDArray[Float]
        The phase ramp evaluated on the normalized pupil mesh, i.e.
        phase(px, py) = tilt_x * px + tilt_y * py.

    """
    tilt_x, tilt_y = tilt_pupil
    normed_coords = np.linspace(-1, 1, mesh_size)
    px, py = np.meshgrid(normed_coords, normed_coords)
    return (tilt_x * px + tilt_y * py).astype(Float)


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
    gaussian_pupil(beam_center_pupil, waist_pupil, mesh_size, polarization)
        Create a Gaussian pupil field with a specified waist size.
    gaussian_halfmoon_pupil(beam_center_pupil, waist_pupil, mesh_size, polarization, orientation, phase, phase_mask_center)
        Create a halfmoon pupil field with a Gaussian beam amplitude.
    uniform_pupil(mesh_size, polarization)
        Create a uniform pupil field with specified polarization.
    with_phase_ramp(tilt_pupil)
        Return a new InputField with a linear phase ramp added to the phase.
    with_zernike_modes(noll_indices, coefficients)
        Return a new InputField with a Zernike phase aberration added to the phase.

    """
    amplitude_x: NDArray[Float]
    amplitude_y: NDArray[Float]
    phase_x: NDArray[Float]
    phase_y: NDArray[Float]
    polarization_x: NDArray[Complex]
    polarization_y: NDArray[Complex]

    @classmethod
    def gaussian_pupil(
        cls,
        beam_center_pupil: tuple[float, float],
        waist_pupil: float | tuple[float, float],
        mesh_size: int,
        polarization: Polarization
    ) -> InputField:
        """Create a Gaussian pupil field with a specified waist size.

        Parameters
        ----------
        beam_center_pupil : tuple of float
            The center of the Gaussian beam in normalized pupil coordinates (x, y).
        waist_pupil : float or tuple of float
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
        amplitude_x, amplitude_y = gaussian_amplitude(beam_center_pupil, waist_pupil, mesh_size)

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
        beam_center_pupil: tuple[float, float],
        waist_pupil: float | tuple[float, float],
        mesh_size: int,
        polarization:Polarization,
        orientation: HalfmoonPhase = HalfmoonPhase.HORIZONTAL,
        phase: float = np.pi,
        phase_mask_center: tuple[float, float] = (0.0, 0.0),
    ) -> InputField:
        """Create a halfmoon pupil field with a Gaussian beam amplitude.
        
        Parameters
        ----------
        beam_center_pupil : tuple of float
            The center of the Gaussian beam in normalized pupil coordinates (x, y).
        waist_pupil : float or tuple of float
            The waist size of the Gaussian beam in normalized pupil coordinates. If a
            single float is provided, it is used for both x and y dimensions.
        mesh_size : int
            The size of the mesh grid for the pupil field.
        polarization : Polarization
            The polarization state of the field.
        orientation : HalfmoonPhase, optional
            The orientation of the halfmoon phase mask. Default is HalfmoonPhase.HORIZONTAL.
        phase : float, optional
            The phase shift applied to the halfmoon mask. Default is np.pi.
        phase_mask_center : tuple of float, optional
            The center of the phase mask in normalized pupil coordinates (x, y). Default is
            (0.0, 0.0).

        Returns
        -------
        InputField
            The input field with Gaussian amplitude and halfmoon phase mask.

        """
        polarization_x, polarization_y = polarization.arrays(mesh_size)
        amplitude_x, amplitude_y = gaussian_amplitude(beam_center_pupil, waist_pupil, mesh_size)

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

    def with_phase_ramp(self, tilt_pupil: tuple[float, float]) -> InputField:
        """Return a new InputField with a linear phase ramp added to the phase.

        Models a blazed-grating-style beam-steering element (e.g. a galvo mirror or
        SLM tilt pattern) as an abstract phase tilt, composable onto any InputField
        regardless of how it was constructed. Mapping this tilt to a physical beam
        displacement requires a separate, system-specific calibration.

        Parameters
        ----------
        tilt_pupil : tuple of float
            Phase tilt in radians at the pupil edge (px=1, py=1) along the pupil's
            x- and y-directions, i.e. (tilt_x, tilt_y). Any combination of tilt_x
            and tilt_y is allowed, giving a ramp in any direction across the pupil
            (not limited to 45 degrees) — e.g. (1.0, 0.0) steers along x, (0.0, 1.0)
            along y, (1.0, 1.0) diagonally. The resulting ramp array is added to
            both of InputField's phase_x and phase_y attributes (its two
            polarization-channel phase arrays), since a physical steering element
            deflects the whole beam rather than one Jones component selectively.

        Returns
        -------
        InputField
            A new instance with the ramp added to phase_x and phase_y. The
            original InputField is not modified.

        """
        ramp = phase_ramp(tilt_pupil, self.phase_x.shape[0])
        return replace(
            self,
            phase_x=self.phase_x + ramp,
            phase_y=self.phase_y + ramp,
        )

    def with_zernike_modes(
        self,
        noll_indices: int | Sequence[int],
        coefficients: float | Sequence[float],
    ) -> InputField:
        """Return a new InputField with a Zernike phase aberration added to the phase.

        Models a wavefront aberration (e.g. optical system aberrations or an SLM
        correction pattern) as a weighted sum of Noll-normalized Zernike polynomials,
        composable onto any InputField regardless of how it was constructed. Requires
        the optional `zernipax` dependency; see the `zernike` extra and
        `leb.just_focus.zernike` for details.

        Parameters
        ----------
        noll_indices : int or sequence of int
            Noll's sequential index (see
            https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices)
            of each Zernike mode to add.
        coefficients : float or sequence of float
            Coefficient in radians for each mode listed in `noll_indices`. Must have
            the same number of elements as `noll_indices`.

        Returns
        -------
        InputField
            A new instance with the Zernike phase added to phase_x and phase_y. The
            original InputField is not modified.

        """
        phase = zernike_phase(noll_indices, coefficients, self.phase_x.shape[0])
        return replace(
            self,
            phase_x=self.phase_x + phase,
            phase_y=self.phase_y + phase,
        )
