# Just Focus

![Tests](https://github.com/LEB-EPFL/just-focus/actions/workflows/tests.yml/badge.svg)

Just Focus is a Python package for computing vectorial electromagnetic fields in the focus of high numerical aperture microscope objectives.

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

- InFocus (MATLAB) [https://github.com/QF06/InFocus](https://github.com/QF06/InFocus)
- Debye Diffraction Code (MATLAB and Python) [https://github.com/jdmanton/debye_diffraction_code](https://github.com/jdmanton/debye_diffraction_code)

## Resources

- I. Herrera and P. A. Quinto-Su, "Simple computer program to calculate arbitrary tightly focused (propagating and evanescent) vector light fields," arXiv:2211.06725 (2022). [https://doi.org/10.48550/arXiv.2211.06725](https://doi.org/10.48550/arXiv.2211.06725).

This manuscript describes the specific numerical implementation of the vectorial field propagation algorithm used here.

- K. M. Douglass, "Coordinate Systems for Modeling Microscope Objectives," (2024). [https://kylemdouglass.com/posts/coordinate-systems-for-modeling-microscope-objectives/](https://kylemdouglass.com/posts/coordinate-systems-for-modeling-microscope-objectives/)

This blog post explains how to set up the various coordinate systems and numerical meshes for evaluating the results of the Richards-Wolf model for high NA objectives.

- M. Leutenegger, R. Rao, R. A. Leitgeb, and T. Lasser. Fast focus field calculations. Opt. Express 14, 11277-11291 (2006). [https://doi.org/10.1364/OE.14.011277](https://doi.org/10.1364/OE.14.011277)

This manuscript was the first to describe the calculation of vectorial focal fields using the fast Fourier transform.

- L. Novotny and B. Hecht, "Principles of Nano-Optics," Cambridge University Press, pp. 56 - 66 (2006). [https://doi.org/10.1017/CBO9780511813535](https://doi.org/10.1017/CBO9780511813535)

Chapter 3 contains the derivation of the field at the focus of an aplanatic lens.
