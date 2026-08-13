from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import StrEnum

from .backend import be
from .dtypes import Array, complex_dtype, float_dtype
from .inputs import InputField
from .focal_fields import FocalField


class Stop(StrEnum):
    """Defines the type of stop used in the pupil.

    Attributes
    ----------
    UNIFORM : str
        A uniform stop, where the pupil is filled completely.
    TANH : str
        A stop defined by a hyperbolic tangent function. This helps to improve accuracy
        in the focal field by gradually reducing the amplitude near the stop's edge.

        See Leutenegger, et al., Opt. Express 14, 11277-91 (2006), eq. 16 for details.
    """
    UNIFORM = "uniform"
    TANH = "tanh"

    def array(self, px: Array, py: Array, radius: float = 1.0) -> Array:
        match self:
            case Stop.UNIFORM:
                return be.astype((px**2 + py**2) <= radius**2, float_dtype())
            case Stop.TANH:
                mesh_size = px.shape[0]
                return 0.5 * (1 + be.tanh(1.5 * mesh_size * (radius - be.sqrt(px**2 + py**2))))
            case _:
                raise ValueError(f"Unsupported stop type: {self}")


@dataclass
class Pupil:
    na: float = 1.4
    wavelength_um: float = 0.532
    refractive_index: float = 1.518
    focal_length_mm: float = 3.3333
    mesh_size: int = 64
    stop: Stop = Stop.UNIFORM
    stop_radius_pupil: float = 1.0

    x_mm: Array = field(init=False, repr=False)
    y_mm: Array = field(init=False, repr=False)
    stop_arr: Array = field(init=False, repr=False)
    stop_radius_mm: float = field(init=False, repr=False)
    kx: Array = field(init=False, repr=False)
    ky: Array = field(init=False, repr=False)
    kz: Array = field(init=False, repr=False)
    k: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.stop_radius_pupil <= 0.0 or self.stop_radius_pupil > 1.0:
            raise ValueError("stop_radius_pupil must be between 0 and 1.")

        normed_coords = be.linspace(-1, 1, self.mesh_size, dtype=float_dtype())

        px, py = be.meshgrid(normed_coords, normed_coords, indexing='xy')
        self.stop_arr = self.stop.array(px, py, self.stop_radius_pupil)

        # Far field coordinate system
        f = self.focal_length_mm / 1e3 # Convert focal length from mm to meters
        x_scaling = f * self.na
        xinf = px * x_scaling
        yinf = py * x_scaling

        # Save pupil coordinates to make plotting easier
        self.x_mm = normed_coords * x_scaling * 1e3
        self.y_mm = normed_coords * x_scaling * 1e3
        self.stop_radius_mm = x_scaling * 1e3

        # Angular spectrum coordinate system
        k0 = 2 * math.pi * 1e6 / self.wavelength_um # Convert wavelength from um to meters
        self.k: float = k0 * self.refractive_index
        self.kx, self.ky = k0 * xinf / f, k0 * yinf / f

        # Set values of kz outside the pupil to 1 to avoid division by zero later
        one = be.asarray(1.0, dtype=float_dtype())
        self.kz = be.sqrt(be.maximum(one, self.k**2 - self.kx**2 - self.ky**2))

    def propgate(self, z_um: float, inputs: InputField, padding_factor: int = 2) -> FocalField:
        """Propagate the input field to the focal plane at distance z.

        Parameters
        ----------
        z_um : float
            The distance to propagate the field in micrometers.
        inputs : InputField
            The input field to propagate.
        padding_factor : int, optional
            The factor by which to pad the input field arrays before propagation. To
            maintain array sizes that are powers of 2, arrays will be padded so that
            their padded shapes are `2**padding_factor * arr.shape[0]` and
            `2**padding_factor * arr.shape[1]`. Default is 2.

        Returns
        -------
        FocalField
            The field in the desired z plane.

        """
        z = z_um * 1e-6  # Convert z from micrometers to meters
        kz_complex = be.astype(self.kz, complex_dtype())
        defocus = be.exp(1j * kz_complex * z)
        kz_root = be.sqrt(self.kz)
        k_transverse_sq = self.kx**2 + self.ky**2

        phase_x_complex = be.exp(1j * be.astype(inputs.phase_x, complex_dtype()))
        phase_y_complex = be.exp(1j * be.astype(inputs.phase_y, complex_dtype()))

        far_field_x = defocus * self.stop_arr * (
            inputs.polarization_x * inputs.amplitude_x * phase_x_complex * (self.ky**2 + self.kx**2 * self.kz / self.k) + \
            inputs.polarization_y * inputs.amplitude_y * phase_y_complex * (-self.kx * self.ky + self.kx * self.ky * self.kz / self.k)
        ) / k_transverse_sq / kz_root
        far_field_y = defocus * self.stop_arr * (
            inputs.polarization_x * inputs.amplitude_x * phase_x_complex * (-self.kx * self.ky + self.kx * self.ky * self.kz / self.k) + \
            inputs.polarization_y * inputs.amplitude_y * phase_y_complex * (self.kx**2 + self.ky**2 * self.kz / self.k)
        ) / k_transverse_sq / kz_root
        far_field_z = defocus * self.stop_arr * (
            inputs.polarization_x * inputs.amplitude_x * phase_x_complex * (-k_transverse_sq * self.kx / self.k) + \
            inputs.polarization_y * inputs.amplitude_y * phase_y_complex * (-k_transverse_sq * self.ky / self.k)
        ) / k_transverse_sq / kz_root

        padding: tuple[tuple[int, int], tuple[int, int]] = self._pad_width(far_field_x.shape, padding_factor)
        far_field_x_padded = be.pad(far_field_x, padding, value=0.0)
        far_field_y_padded = be.pad(far_field_y, padding, value=0.0)
        far_field_z_padded = be.pad(far_field_z, padding, value=0.0)

        # Compute the phase correction due to not sampling the origin
        # See Herrera and Quinto-Su, arxiv:2211.06725 (2022), section
        # Optical System: Mesh and note the typo in the first minus sign (it should be
        # a plus sign).
        auxillary_mesh_size = self.mesh_size * 2**padding_factor
        auxillary_coords = be.linspace(
            -auxillary_mesh_size // 2, auxillary_mesh_size // 2, auxillary_mesh_size, dtype=float_dtype()
        )
        px, py = be.meshgrid(auxillary_coords, auxillary_coords, indexing='xy')
        correction_scaling = 1j * 2 * math.pi * 0.5 / px.shape[0]
        phase_correction = be.exp(
            correction_scaling * be.astype(px, complex_dtype()) + correction_scaling * be.astype(py, complex_dtype())
        )

        field_x = be.fft.ifftshift(be.fft.ifft2(be.fft.ifftshift(far_field_x_padded))) * phase_correction
        field_y = be.fft.ifftshift(be.fft.ifft2(be.fft.ifftshift(far_field_y_padded))) * phase_correction
        field_z = be.fft.ifftshift(be.fft.ifft2(be.fft.ifftshift(far_field_z_padded))) * phase_correction

        dx = self.wavelength_um / 2 / self.na / 2**padding_factor
        dy = dx
        x_um = dx * be.linspace(-field_x.shape[0] // 2, field_x.shape[0] // 2 - 1, field_x.shape[0], dtype=float_dtype())
        y_um = dy * be.linspace(-field_y.shape[1] // 2, field_y.shape[1] // 2 - 1, field_y.shape[1], dtype=float_dtype())

        return FocalField(field_x=field_x, field_y=field_y, field_z=field_z, x_um=x_um, y_um=y_um)

    @staticmethod
    def _pad_width(array_shape: tuple[int, int], padding_factor: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate the padding width for an array."""
        padded_shape = (2**padding_factor * array_shape[0], 2**padding_factor * array_shape[1])
        pad_height = (padded_shape[0] - array_shape[0]) // 2
        pad_width = (padded_shape[1] - array_shape[1]) // 2
        return ((pad_height, pad_height), (pad_width, pad_width))
