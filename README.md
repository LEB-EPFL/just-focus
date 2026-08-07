# Just Focus

![CI](https://github.com/LEB-EPFL/just-focus/actions/workflows/tests.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/just-focus)
[![DOI](https://zenodo.org/badge/1028418053.svg)](https://zenodo.org/badge/latestdoi/1028418053)

Just Focus is a Python package for computing vectorial electromagnetic fields in the focus of high numerical aperture microscope objectives.

## Quickstart

Compute the field in the focal plane (z = 0.0) of a NA 1.4 oil immersion microscope objective assuming a linearly polarized, paraxial Gaussian beam with a waist size equal to the radius of the objective's back aperture. Use a hyperbolic tangent function to smooth the boundary of the stop and zero pad the mesh so that the final square mesh has 64 * 2^4 = 1024 samples in each direction.

```python
import numpy as np

from leb.just_focus import InputField, Polarization, Pupil, Stop

mesh_size = 64

inputs = InputField.gaussian_pupil(
    beam_center_pupil=(0.0, 0.0),
    waist_pupil=1.0,
    mesh_size=mesh_size,
    polarization=Polarization.LINEAR_Y,
)

pupil = Pupil(
    na=1.4,
    refractive_index=1.518,
    wavelength_um=0.561,
    mesh_size=mesh_size,
    stop=Stop.TANH,
)

results = pupil.propgate(0.0, inputs, padding_factor=4)
```

## Installation

```console
pip install just-focus
```

### Extras

Install additional dependencies for making plots:

```console
pip install just-focus[plot]
```

Then you can use functions in the `leb.just_focus.plots` module to plot the inputs and results.

```python
from leb.just_focus.plots import plot_inputs

plot_inputs(inputs, pupil)

```

## Use

just-focus follows this workflow:

1. Define your input field in the pupil using `InputField`.
2. Define a pupil using `Pupil`.
3. Compute the focal field in the desired z-plane using the `Pupil.propagate` method.

`Pupil.propagate` returns an instance of a `FocalField` object which contains a complex 2D array for each field direction.

### InputField

Six parameters are required to construct a new `InputField`:

```python
from leb.just_focus import InputField

input = InputField(
    amplitude_x,
    amplitude_y,
    phase_x,
    phase_y,
    polarization_x,
    polarization_y,
)
```

All parameters should be 2D square arrays whose shape elements are powers of 2. The amplitude and phase arrays are of dtype `np.float64`, and the polarization arrays are of dtype `np.complex128`.

These inputs follow the implementation laid out by [Herrera and Quinto-Su](https://doi.org/10.48550/arXiv.2211.06725). Technically, they overspecify the field at the pupil in many "normal" cases. They are all required, however, to model a beam-shaping experiment where the x- and y-components of the field may be independently modulated in amplitude, phase, and polarization, such as setups with two SLMs and polarizing elements on two separate beam paths.

If all you want is to specify the amplitude and phase of the x- and y-components of the field at the pupil independently, set each of `polarization_x` and `polarization_y` to all ones. The elements of the resulting Jones vector describing the polarization at a point (x, y) in the pupil are then:

```
E_x = A_x / sqrt(A_x^2 + A_y^2)
E_y = A_y * exp(1j * (phi_y - phi_x)) / sqrt(A_x^2 + A_y^2)
```

where `A_x, A_y, phi_x, phi_y` are the amplitudes and phases in the x and y directions, respectively.

Alternatively, the relative phases may be determined by setting `phase_x` and `phase_y` to all zeros and setting the polarization arrays accordingly.

#### Common Input Fields

Some factory methods exist to compute commonly encountered input fields:

```python
from leb.just_focus import HalfmoonPhase, InputField, Polarization

mesh_size = 64

gaussian = InputField.gaussian_pupil(
    beam_center_pupil=(0.0, 0.0),
    waist_pupil=1.0,
    mesh_size=mesh_size,
    polarization=Polarization.LINEAR_Y,
)

halfmoon = InputField.gaussian_halfmoon_pupil(
    beam_center_pupil=(0.0, 0.5),
    waist_pupil=2.0,
    mesh_size=mesh_size,
    polarization=Polarization.LINEAR_Y,
    orientation=HalfmoonPhase.MINUS_45,
    phase=np.pi,
    phase_mask_center=(0.0, 0.0),
)

uniform = InputField.uniform_pupil(
    mesh_size=mesh_size,
    polarization=Polarization.CIRCULAR_LEFT,
)
```

Coordinates and waist sizes are in units of normalized pupil coordinates, i.e. 0 is at the center and 1 is at the pupil edge.

Possible values for the `Polarization` enum are:

```python
Polarization.LINEAR_X
Polarization.LINEAR_Y
Polarization.CIRCULAR_LEFT
Polarization.CIRCULAR_RIGHT
```

Possible values for the `HalfmoonPhase` enum are:

```python
HalfmoonPhase.HORIZONTAL
HalfmoonPhase.VERTICAL
HalfmoonPhase.MINUS_45
HalfmoonPhase.PLUS_45
```

#### Beam Steering with a Phase Ramp

A linear phase ramp (blazed grating) can be composed onto any `InputField`, regardless of how it was constructed, to model beam-steering elements such as galvo mirrors or SLM tilt patterns:

```python
steered = halfmoon.with_phase_ramp(tilt_pupil=(0.5, 0.0))
```

`tilt_pupil` specifies the phase tilt in radians at the pupil edge (`px=1`/`py=1`) along the x- and y-directions, and may point in any direction, e.g. `(1.0, 0.0)` steers along x, `(0.0, 1.0)` along y, `(1.0, 1.0)` diagonally.

See `scripts/displaced_gaussian.py` for a runnable example that steers a focused Gaussian beam with `tilt_pupil=(-2.0, 1.0)` and plots the resulting displacement (requires the `plot` extra):

```console
uv run displaced_gaussian
```

### Pupil

A `Pupil` instance is defined as follows:

```python
from leb.just_focus import Pupil, Stop

pupil = Pupil(
    na=1.4,
    wavelength_um=0.561,
    refractive_index=1.518,
    focal_length_mm=3.3333,
    mesh_size=64,
    stop=Stop.TANH,
    stop_radius_pupil=1.0,
)
```

The refractive index is that of the immersion medium. The incident beam is assumed to be incident from air (n = 1).

The focal length of an objective may be computed from the ratio between the corresponding tube lens focal length and its magnification. For example, a 100x Nikon objective will have a focal length of 2 mm because Nikon tube lenses have focal lengths of 200 mm, and 200 mm / 100 = 2 mm. The focal length used here is the focal length of the objective for a sample in air, i.e. the distance from the principle plane where the paraxial marginal ray from an object located at infinity intersects the optical axis in air. It is not already multiplied by the refractive index of the immersion medium, which is the convention used in Herrera and Quinto-Su and the textbook by Novotny and Hecht. The convention used in this package puts the location of the focus at a distance `n * f` from the principle reference sphere in sample space. This is consistent with the well-known formula `R = f * NA` for the radius of the back aperture of the objective. See the Resources section below for more information.

The stop parameter determines whether and how the aperture should be softened to reduce artifacts from the fast Fourier transform. Possible values are:

```
Stop.UNIFORM
Stop.TANH
```

A uniform stop is a pupil with a discontinuous edge. `Stop.TANH` softens this edge with a hyperbolic tangent function as introduced by Leutenegger, et al. in the Resources section below.

`stop_radius_pupil` sets the radius of the stop in normalized pupil coordinates (1.0 is the pupil's edge, i.e. the rated NA). Values less than 1.0 model stopping down the pupil, e.g. with an iris, while keeping `na` fixed; the input field is simply cropped to this radius before propagation.

#### Pupil.propagate

To compute the focal field at a given z plane, use:

```python
pupil.propagate(z_um, inputs, padding_factor=4)
```

where `z_um = 0` corresponds to the focal plane of the objective and`inputs` is an `InputField` instance.

`padding_factor` describes the amount by which the input field will be zero-padded before computing the fast Fourier transforms. If the linear size of an input field array is N, then the padded array will be of size `N * 2^padding_factor` in each dimension. This will also be the size of the resulting focal field arrays.

### FocalField

`Pupil.propagate` returns a `FocalField` instance which is defined as follows:

```python
@dataclass(frozen=True)
class FocalField:
    field_x: NDArray[Complex]
    field_y: NDArray[Complex]
    field_z: NDArray[Complex]
    x_um: NDArray[Float]
    y_um: NDArray[Float]

    def intensity(self, normalize: bool = True) -> NDArray[Float]:
        I = np.abs(self.field_x)**2 + np.abs(self.field_y)**2 + np.abs(self.field_z)**2
        if normalize:
            return I / np.max(I)
        return I
```

It has five parameters: three, 2D complex arrays representing the field in each direction and two, 1D arrays representing the x- and y-coordinates in the focal region.

In addition, there is an `intenstiy` helper method that computes the intensity from the fields.

### Coordinate Reference Systems and Meshes

There are two, 2D computational meshes used in just-focus:

1. the pupil mesh, and
2. the focal field mesh.

The pupil mesh has two different coordinate reference systems: one for the real physical coordinates of the pupil and another for the k-space coordinates. The only difference between the two is that the physical mesh is scaled by the objective focal length (in air) times the NA, whereas the k-space mesh is scaled by the free space wavevector times the NA.

![An illustration of the coordinate system and the computational mesh used in these simulations.](/assets/pupil-function-simulation-mesh.png)

The pupil mesh samples are always taken at the centers of their corresponding cells. The origin is at the corners where the four center cells meet; as a result, the origin of the pupil is not sampled, which is useful for avoiding divisions by zero during field calculations. On the other hand, by not sampling the origin the code must apply a phase correction term to the samples in k-space to ensure correct application of the FFT. (See the manuscript by Herrera and Quinto-Su cited below for more information.)

Unlike the pupil mesh, the origin of the coordinate system is sampled by the focal field mesh because of how the FFT works. It lies at pixel `L / 2`, where `L` is the linear square mesh size. The focal field mesh spacing is `dx = λ/(2·NA·2^padding_factor)` and total the FOV is `L * dx = mesh_size * λ/(2·NA)`, i.e. the FOV is independent of any padding applied before the FFT.

### Example Scripts

Command line scripts that illustrate the use of Just Focus may be found in [src/leb/just_focus/scripts](src/leb/just_focus/scripts). They are also available on the command line, i.e. `uv run gaussian`.

Scripts require the `plot` set of optional dependencies. See [the installation instructions](#extras) for more details about how to install them.

## Development

### Set up the development environment

Development requires [uv](https://docs.astral.sh/uv/).

After cloning this repo, run the following command from the project's root directory:

```console
uv sync --all-extras
```

This will create a virtual environment with the required dependencies in a folder named `.venv`.

### Tests

Just run `pytest` from the project's root directory:

```console
pytest
```

## Other Packages to Compute Vectorial Focal Fields

- PSF-Generator (Python) <https://github.com/Biomedical-Imaging-Group/psf_generator>
- InFocus (MATLAB) <https://github.com/QF06/InFocus>
- Debye Diffraction Code (MATLAB and Python) <https://github.com/jdmanton/debye_diffraction_code>
- PyFocus <https://github.com/fcaprile/PyFocus>
- PSF Generator (Java) <https://bigwww.epfl.ch/algorithms/psfgenerator/>

### Is Just Focus for me?

- If you want a fast PSF calculator that runs on the GPU, then use [psf-generator](https://github.com/Biomedical-Imaging-Group/psf_generator).
- If you want a GUI and/or a Windows-installable executable, then use [PyFocus](https://github.com/fcaprile/PyFocus).
- If you want a MATLAB tool, then use [InFocus](https://github.com/QF06/InFocus).
- If you want a Java/ImageJ/Fiji/Icy tool, use [PSF Generator](https://bigwww.epfl.ch/algorithms/psfgenerator/).

If you want 

1. a Python package
2. that computes vectorial focal fields
2. with a simple API and
3. a small number of dependencies,
3. you do not care about computation speed or optimizations, and
4. you want the physics clearly reflected in the code,

then Just Focus might be for you.

## Resources

- I. Herrera and P. A. Quinto-Su, "Simple computer program to calculate arbitrary tightly focused (propagating and evanescent) vector light fields," arXiv:2211.06725 (2022). [https://doi.org/10.48550/arXiv.2211.06725](https://doi.org/10.48550/arXiv.2211.06725).

This manuscript describes the specific numerical implementation of the vectorial field propagation algorithm used here.

- K. M. Douglass, "Coordinate Systems for Modeling Microscope Objectives," (2024). [https://kylemdouglass.com/posts/coordinate-systems-for-modeling-microscope-objectives/](https://kylemdouglass.com/posts/coordinate-systems-for-modeling-microscope-objectives/)

This blog post explains how to set up the various coordinate systems and numerical meshes for evaluating the results of the Richards-Wolf model for high NA objectives.

- M. Leutenegger, R. Rao, R. A. Leitgeb, and T. Lasser. Fast focus field calculations. Opt. Express 14, 11277-11291 (2006). [https://doi.org/10.1364/OE.14.011277](https://doi.org/10.1364/OE.14.011277)

This manuscript was the first to describe the calculation of vectorial focal fields using the fast Fourier transform.

- L. Novotny and B. Hecht, "Principles of Nano-Optics," Cambridge University Press, pp. 56 - 66 (2006). [https://doi.org/10.1017/CBO9780511813535](https://doi.org/10.1017/CBO9780511813535)

Chapter 3 contains the derivation of the field at the focus of an aplanatic lens.
