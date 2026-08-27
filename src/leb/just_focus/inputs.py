"""Input fields for the propagation algorithm."""

from __future__ import annotations
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from .backend import be
from .dtypes import Array, complex_dtype, float_dtype
from .zernike import _zernike_basis


class Polarization(StrEnum):
    LINEAR_X = "linear_x"
    LINEAR_Y = "linear_y"
    LINEAR_PLUS_45 = "linear_plus_45"
    LINEAR_MINUS_45 = "linear_minus_45"
    CIRCULAR_LEFT = "circular_left"
    CIRCULAR_RIGHT = "circular_right"

    def arrays(self, mesh_size: int) -> tuple[Array, Array]:
        match self:
            case Polarization.LINEAR_X:
                polarization_x = be.ones((mesh_size, mesh_size), dtype=complex_dtype())
                polarization_y = be.zeros((mesh_size, mesh_size), dtype=complex_dtype())
            case Polarization.LINEAR_Y:
                polarization_x = be.zeros((mesh_size, mesh_size), dtype=complex_dtype())
                polarization_y = be.ones((mesh_size, mesh_size), dtype=complex_dtype())
            case Polarization.LINEAR_PLUS_45:
                polarization_x = be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
                polarization_y = be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
            case Polarization.LINEAR_MINUS_45:
                polarization_x = be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
                polarization_y = -be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
            case Polarization.CIRCULAR_LEFT:
                polarization_x = be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
                polarization_y = 1j * be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
            case Polarization.CIRCULAR_RIGHT:
                polarization_x = be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
                polarization_y = -1j * be.ones((mesh_size, mesh_size), dtype=complex_dtype()) / math.sqrt(2)
            case _:
                raise ValueError(f"Unsupported polarization: {self}")

        return polarization_x, polarization_y


class HalfmoonPhase(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    PLUS_45 = "plus_45"
    MINUS_45 = "minus_45"

    def arrays(
        self,
        mesh_size: int,
        phase: float = math.pi,
        phase_mask_center: tuple[float, float] = (0.0, 0.0)
    ) -> tuple[Array, Array]:
        normed_coords = be.linspace(-1, 1, mesh_size, dtype=float_dtype())
        x, y = be.meshgrid(normed_coords, normed_coords, indexing='xy')
        x0, y0 = phase_mask_center
        x = x - x0
        y = y - y0

        phase_val = be.asarray(phase, dtype=float_dtype())
        zero_val = be.asarray(0.0, dtype=float_dtype())
        match self:
            case HalfmoonPhase.HORIZONTAL:
                mask = x >= 0
            case HalfmoonPhase.VERTICAL:
                mask = y >= 0
            case HalfmoonPhase.PLUS_45:
                mask = (x + y) >= 0
            case HalfmoonPhase.MINUS_45:
                mask = (x - y) >= 0
            case _:
                raise ValueError(f"Unsupported halfmoon orientation: {self}")

        phase_x = be.where(mask, phase_val, zero_val)
        phase_y = be.copy(phase_x)

        return phase_x, phase_y


def gaussian_amplitude(
    beam_center_pupil: tuple[float, float],
    waist_pupil: float | tuple[float, float],
    mesh_size: int
) -> tuple[Array, Array]:
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
    tuple of Array
        The Gaussian amplitude for the x- and y-directions, respectively.

    """
    if isinstance(waist_pupil, (int, float)):
        waist_x = waist_y = waist_pupil
    else:
        waist_x, waist_y = waist_pupil

    normed_coords = be.linspace(-1, 1, mesh_size, dtype=float_dtype())
    x, y = be.meshgrid(normed_coords, normed_coords, indexing='xy')
    x0: float = beam_center_pupil[0]
    y0: float = beam_center_pupil[1]
    amplitude_x = be.exp(-(x - x0)**2 / waist_x**2 - (y - y0)**2 / waist_y**2)
    amplitude_y = be.copy(amplitude_x)

    return amplitude_x, amplitude_y


def phase_ramp(tilt_pupil: tuple[float, float], mesh_size: int) -> Array:
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
    Array
        The phase ramp evaluated on the normalized pupil mesh, i.e.
        phase(px, py) = tilt_x * px + tilt_y * py.

    """
    tilt_x, tilt_y = tilt_pupil
    normed_coords = be.linspace(-1, 1, mesh_size, dtype=float_dtype())
    px, py = be.meshgrid(normed_coords, normed_coords, indexing='xy')
    return be.astype(tilt_x * px + tilt_y * py, float_dtype())


@dataclass
class InputField:
    """Factory class for creating input fields for the pupil.

    Each direction may be specified independently, which models separate beam shaping
    elements for the x- and y-directions. In many common cases, the amplitudes and
    phases will be the same in both x- and y-directions and only the polarization will
    differ.

    Attributes
    ----------
    amplitude_x : Array
        The amplitude of the field for the x-direction.
    amplitude_y : Array
        The amplitude of the field for the y-direction.
    phase_x : Array
        The phase of the field for the x-direction.
    phase_y : Array
        The phase of the field for the y-direction.
    polarization_x : Array
        The polarization state of the field for the x-direction.
    polarization_y : Array
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
    amplitude_x: Array
    amplitude_y: Array
    phase_x: Array
    phase_y: Array
    polarization_x: Array
    polarization_y: Array

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

        phase_x = be.zeros((mesh_size, mesh_size), dtype=float_dtype())
        phase_y = be.zeros((mesh_size, mesh_size), dtype=float_dtype())

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
        phase: float = math.pi,
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
            The phase shift applied to the halfmoon mask. Default is pi.
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

        amplitude_x = be.ones((mesh_size, mesh_size), dtype=float_dtype())
        amplitude_y = be.ones((mesh_size, mesh_size), dtype=float_dtype())
        phase_x = be.zeros((mesh_size, mesh_size), dtype=float_dtype())
        phase_y = be.zeros((mesh_size, mesh_size), dtype=float_dtype())

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
        coefficients: float | Sequence[float] | Array,
    ) -> InputField:
        """Return a new InputField with a Zernike phase aberration added to the phase.

        Models a wavefront aberration (e.g. optical system aberrations or an SLM
        correction pattern) as a weighted sum of Noll-normalized Zernike polynomials,
        composable onto any InputField regardless of how it was constructed. Requires
        the optional `zernipax` dependency; see the `zernike` extra and
        `leb.just_focus.zernike` for details.

        The mode basis (fixed once `noll_indices` and the mesh size are known) is
        still built via `zernipax` (NumPy/JAX) and cached; it carries no autograd
        history. The combination with `coefficients` happens natively in the active
        backend, though, so under the PyTorch backend a `coefficients` tensor with
        `requires_grad=True` (e.g. a Pyro latent site) keeps its autograd graph
        through this call, i.e. gradients flow into the coefficients, just not through
        the basis construction itself.

        Parameters
        ----------
        noll_indices : int or sequence of int
            Noll's sequential index (see
            https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices)
            of each Zernike mode to add.
        coefficients : float, sequence of float, or Array
            Coefficient in radians for each mode listed in `noll_indices`. Must have
            the same number of elements as `noll_indices`. Under the PyTorch backend,
            may be a tensor (e.g. `requires_grad=True`) whose autograd graph should
            be preserved.

        Returns
        -------
        InputField
            A new instance with the Zernike phase added to phase_x and phase_y. The
            original InputField is not modified.

        """
        mesh_size = self.phase_x.shape[0]
        basis = _zernike_basis(noll_indices, mesh_size)
        coefficients_backend = be.atleast_1d(be.asarray(coefficients, dtype=float_dtype()))
        if basis.shape[1] != coefficients_backend.shape[0]:
            raise ValueError(
                "noll_indices and coefficients must have the same number of elements, "
                f"got {basis.shape[1]} and {coefficients_backend.shape[0]}."
            )

        basis_backend = be.asarray(basis, dtype=float_dtype())
        phase = (basis_backend @ coefficients_backend).reshape(mesh_size, mesh_size)
        return replace(
            self,
            phase_x=self.phase_x + phase,
            phase_y=self.phase_y + phase,
        )
