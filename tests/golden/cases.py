"""Shared registry of pipeline configurations used as golden regression data.

Imported by both `generate_golden_data.py` (which produces the frozen
`.npz` files under `data/`) and `test_backend_parity.py` (which rebuilds the
same inputs and compares against those files), so the two definitions can
never drift apart.

Each case is built fresh by its builder function so that constructing it
under whatever array backend is currently active (see
`leb.just_focus.backend`) picks up that backend's arrays/tensors.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass

from leb.just_focus import HalfmoonPhase, InputField, Polarization, Pupil, Stop

HAS_ZERNIPAX = importlib.util.find_spec("zernipax") is not None

MESH_SIZE = 16
PADDING_FACTOR = 3


@dataclass(frozen=True)
class GoldenCase:
    name: str
    pupil: Pupil
    inputs: InputField
    z_um: float


def _uniform_linear_x() -> GoldenCase:
    return GoldenCase(
        name="uniform_linear_x",
        pupil=Pupil(mesh_size=MESH_SIZE),
        inputs=InputField.uniform_pupil(MESH_SIZE, Polarization.LINEAR_X),
        z_um=0.0,
    )


def _uniform_circular_defocus() -> GoldenCase:
    return GoldenCase(
        name="uniform_circular_defocus",
        pupil=Pupil(mesh_size=MESH_SIZE),
        inputs=InputField.uniform_pupil(MESH_SIZE, Polarization.CIRCULAR_LEFT),
        z_um=0.3,
    )


def _gaussian_offcenter_tanh_stop() -> GoldenCase:
    return GoldenCase(
        name="gaussian_offcenter_tanh_stop",
        pupil=Pupil(mesh_size=MESH_SIZE, stop=Stop.TANH, stop_radius_pupil=0.9),
        inputs=InputField.gaussian_pupil(
            (0.3, 0.1), 0.7, MESH_SIZE, Polarization.LINEAR_Y
        ),
        z_um=0.1,
    )


def _halfmoon_vertical() -> GoldenCase:
    return GoldenCase(
        name="halfmoon_vertical",
        pupil=Pupil(mesh_size=MESH_SIZE),
        inputs=InputField.gaussian_halfmoon_pupil(
            (0.0, 0.0),
            0.7,
            MESH_SIZE,
            Polarization.LINEAR_X,
            orientation=HalfmoonPhase.VERTICAL,
            phase=math.pi / 2,
        ),
        z_um=0.0,
    )


def _phase_ramp_tilt() -> GoldenCase:
    inputs = InputField.uniform_pupil(MESH_SIZE, Polarization.LINEAR_X).with_phase_ramp(
        (0.5, -0.3)
    )
    return GoldenCase(
        name="phase_ramp_tilt",
        pupil=Pupil(mesh_size=MESH_SIZE),
        inputs=inputs,
        z_um=0.2,
    )


def _zernike_astigmatism() -> GoldenCase:
    inputs = InputField.uniform_pupil(MESH_SIZE, Polarization.LINEAR_X).with_zernike_modes(
        [4, 6], [0.3, 0.2]
    )
    return GoldenCase(
        name="zernike_astigmatism",
        pupil=Pupil(mesh_size=MESH_SIZE),
        inputs=inputs,
        z_um=0.0,
    )


_BUILDERS = {
    "uniform_linear_x": _uniform_linear_x,
    "uniform_circular_defocus": _uniform_circular_defocus,
    "gaussian_offcenter_tanh_stop": _gaussian_offcenter_tanh_stop,
    "halfmoon_vertical": _halfmoon_vertical,
    "phase_ramp_tilt": _phase_ramp_tilt,
}
if HAS_ZERNIPAX:
    _BUILDERS["zernike_astigmatism"] = _zernike_astigmatism


def case_names() -> list[str]:
    """Names of every registered golden case."""
    return list(_BUILDERS)


def build_case(name: str) -> GoldenCase:
    """Build (fresh) the golden case registered under `name`."""
    return _BUILDERS[name]()


def all_cases() -> list[GoldenCase]:
    """Build (fresh) every registered golden case."""
    return [builder() for builder in _BUILDERS.values()]
