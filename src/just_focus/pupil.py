from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.fft import fftshift, ifft2, ifftshift


Complex = np.complex128
Float = np.float64


class Polarization(StrEnum):
    LINEAR_X = "linear_x"
    LINEAR_Y = "linear_y"
    CIRCULAR_LEFT = "circular_left"
    CIRCULAR_RIGHT = "circular_right"


class HalfmoonOrientation(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    PLUS_45 = "plus_45"
    MINUS_45 = "minus_45"


@dataclass
class InputField:
    """Factory class for creating input fields for the pupil.

    Each direction may be specified independently, which models separate beam shaping
    elements for the x- and y-directions. In many common cases, the amplitudes and
    phases will be the same in both x- and y-directions and only the polarization will
    differ.
    
    Attributes
    ----------
    amplitude_x : np.ndarray
        The amplitude of the field for the x-direction.
    amplitude_y : np.ndarray
        The amplitude of the field for the y-direction.
    phase_x : np.ndarray
        The phase of the field for the x-direction.
    phase_y : np.ndarray
        The phase of the field for the y-direction.
    polarization_x : np.ndarray
        The polarization state of the field for the x-direction.
    polarization_y : np.ndarray
        The polarization state of the field for the y-direction.

    Methods
    -------
    gaussian_pupil(beam_center, waist, mesh_size, polarization)
        Create a Gaussian pupil field with a specified waist size.
    uniform_pupil(mesh_size, polarization)
        Create a uniform pupil field with specified polarization.

    """
    amplitude_x: np.ndarray
    amplitude_y: np.ndarray
    phase_x: np.ndarray
    phase_y: np.ndarray
    polarization_x: np.ndarray
    polarization_y: np.ndarray

    @staticmethod
    def _gaussian_amplitude(
        beam_center: tuple[float, float],
        waist: float | tuple[float, float],
        mesh_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate a Gaussian amplitude for the pupil field."""
        if isinstance(waist, (int, float)):
            waist_x = waist_y = waist
        else:
            waist_x, waist_y = waist

        normed_coords = np.linspace(-1, 1, mesh_size)
        x, y = np.meshgrid(normed_coords, normed_coords)

        amplitude_x = np.exp(-(x - beam_center[0])**2 / waist_x**2 - (y - beam_center[1])**2 / waist_y**2)
        amplitude_y = np.copy(amplitude_x)

        return amplitude_x, amplitude_y

    @staticmethod
    def _uniform_polarization(mesh_size: int, polarization: Polarization) -> tuple[np.ndarray, np.ndarray]:
        match polarization:
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

    @classmethod
    def gaussian_pupil(
        cls,
        beam_center: tuple[float, float],
        waist: float | tuple[float, float],
        mesh_size: int,
        polarization:Polarization
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
        polarization_x, polarization_y = cls._uniform_polarization(mesh_size, polarization)
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
        orientation: HalfmoonOrientation = HalfmoonOrientation.HORIZONTAL,
        phase: float = np.pi,
    ) -> InputField:
        polarization_x, polarization_y = cls._uniform_polarization(mesh_size, polarization)
        amplitude_x, amplitude_y = cls._gaussian_amplitude(beam_center, waist, mesh_size)

        normed_coords = np.linspace(-1, 1, mesh_size)
        x, y = np.meshgrid(normed_coords, normed_coords)

        phase_x = np.zeros((mesh_size, mesh_size), dtype=Float)
        match orientation:
            case HalfmoonOrientation.HORIZONTAL:
                phase_x[x >= 0] = phase
            case HalfmoonOrientation.VERTICAL:
                phase_x[y >= 0] = phase
            case HalfmoonOrientation.PLUS_45:
                phase_x[(x + y) >= 0] = phase
            case HalfmoonOrientation.MINUS_45:
                phase_x[(x - y) >= 0] = phase

        phase_y = phase_x.copy()

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
        polarization_x, polarization_y = cls._uniform_polarization(mesh_size, polarization)
            
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


@dataclass(frozen=True)
class FocalField:
    field_x: np.ndarray
    field_y: np.ndarray
    field_z: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray

    def intensity(self, normalize: bool = True) -> np.ndarray:
        I = np.abs(self.field_x)**2 + np.abs(self.field_y)**2 + np.abs(self.field_z)**2
        if normalize:
            return I / np.max(I)
        return I

@dataclass
class Pupil:
    na: float = 1.4
    wavelength_um: float = 0.532
    refractive_index: float = 1.518
    focal_length_mm: float = 3.3333
    mesh_size: int = 64

    x_mm: np.ndarray = field(init=False, repr=False)
    y_mm: np.ndarray = field(init=False, repr=False)
    stop: np.ndarray = field(init=False, repr=False)
    stop_radius_mm: float = field(init=False, repr=False)
    kx: np.ndarray = field(init=False, repr=False)
    ky: np.ndarray = field(init=False, repr=False)
    kz: np.ndarray = field(init=False, repr=False)
    k: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normed_coords = np.linspace(-1, 1, self.mesh_size)
        
        px, py = np.meshgrid(normed_coords, normed_coords)
        self.stop = ((px**2 + py**2) <= 1).astype(Float)

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
        k0 = 2 * np.pi * 1e6 / self.wavelength_um # Convert wavelength from um to meters
        self.k = k0 * self.refractive_index
        self.kx, self.ky = k0 * xinf / f, k0 * yinf / f

        # Set values of kz outside the pupil to 1 to avoid division by zero later
        self.kz = np.sqrt(np.maximum(1, self.k**2 - self.kx**2 - self.ky**2))

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
        defocus = np.exp(1j * self.kz * z)
        kz_root = np.sqrt(self.kz)
        k_transverse_sq = self.kx**2 + self.ky**2

        far_field_x = defocus * self.stop * (
            inputs.polarization_x * inputs.amplitude_x * np.exp(1j * inputs.phase_x) * (self.ky**2 + self.kx**2 * self.kz / self.k) + \
            inputs.polarization_y * inputs.amplitude_y * np.exp(1j * inputs.phase_y) * (-self.kx * self.ky + self.kx * self.ky * self.kz / self.k)
        ) / k_transverse_sq / kz_root
        far_field_y = defocus * self.stop * (
            inputs.polarization_x * inputs.amplitude_x * np.exp(1j * inputs.phase_x) * (-self.kx * self.ky + self.kx * self.ky * self.kz / self.k) + \
            inputs.polarization_y * inputs.amplitude_y * np.exp(1j * inputs.phase_y) * (self.kx**2 + self.ky**2 * self.kz / self.k)
        ) / k_transverse_sq / kz_root
        far_field_z = defocus * self.stop * (
            inputs.polarization_x * inputs.amplitude_x * np.exp(1j * inputs.phase_x) * (-k_transverse_sq * self.kx / self.k) + \
            inputs.polarization_y * inputs.amplitude_y * np.exp(1j * inputs.phase_y) * (-k_transverse_sq * self.ky / self.k)
        ) / k_transverse_sq / kz_root

        padding: tuple[tuple[int, int], tuple[int, int]] = self._pad_width(far_field_x.shape, padding_factor)
        far_field_x_padded = np.pad(far_field_x, padding, mode='constant', constant_values=0)
        far_field_y_padded = np.pad(far_field_y, padding, mode='constant', constant_values=0)
        far_field_z_padded = np.pad(far_field_z, padding, mode='constant', constant_values=0)

        field_x = fftshift(ifft2(ifftshift(far_field_x_padded)))
        field_y = fftshift(ifft2(ifftshift(far_field_y_padded)))
        field_z = fftshift(ifft2(ifftshift(far_field_z_padded)))

        dx = self.wavelength_um / 2 / self.na / 2**padding_factor
        dy = dx
        x_um = np.linspace(-dx * (field_x.shape[0] // 2), dx * (field_x.shape[0] // 2), field_x.shape[0])
        y_um= np.linspace(-dy * (field_x.shape[1] // 2), dy * (field_y.shape[1] // 2), field_y.shape[1])

        return FocalField(field_x=field_x, field_y=field_y, field_z=field_z, x_um=x_um, y_um=y_um)
    
    @staticmethod
    def _pad_width(array_shape: tuple[int, int], padding_factor: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Calculate the padding width for an array."""
        padded_shape = (2**padding_factor * array_shape[0], 2**padding_factor * array_shape[1])
        pad_height = (padded_shape[0] - array_shape[0]) // 2
        pad_width = (padded_shape[1] - array_shape[1]) // 2
        return ((pad_height, pad_height), (pad_width, pad_width))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    np.seterr("raise")

    mesh_size = 32
    pupil = Pupil(
        na=1.4,
        refractive_index=1.518,
        wavelength_um=0.561,
        mesh_size=mesh_size
    )
    # inputs = InputField.uniform_pupil(mesh_size, Polarization.CIRCULAR_LEFT)
    
    # inputs = InputField.gaussian_pupil(
    #     beam_center=(0.0, 0.0),
    #     waist=2.0,
    #     mesh_size=mesh_size,
    #     polarization=Polarization.CIRCULAR_LEFT
    # )

    inputs = InputField.gaussian_halfmoon_pupil(
        beam_center=(0.5, 0.5),
        waist=2.0,
        mesh_size=mesh_size,
        polarization=Polarization.CIRCULAR_LEFT,
        orientation=HalfmoonOrientation.MINUS_45,
        phase=np.pi
    )

    results = pupil.propgate(0.0, inputs, padding_factor=5)

    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
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

    axs[1, 2].imshow(pupil.stop, vmin=0, vmax=1)
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
    axs[1, 3].set_xlabel("x, $\mu m$")
    axs[1, 3].set_xlim(-1, 1)
    axs[1, 3].set_ylim(-1, 1)
    plt.show()
