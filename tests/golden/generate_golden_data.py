"""One-off script: (re)generate the golden regression data under `data/`.

Run manually via `uv run python tests/golden/generate_golden_data.py` from the
repo root and only against a version of `leb.just_focus` believed correct.
Never run this automatically (it is not collected by pytest) and never
re-run it casually after changing `pupil.py`/`inputs.py`: these `.npz` files
are the frozen golden data that `test_backend_parity.py` checks both the
NumPy and PyTorch code paths against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cases import PADDING_FACTOR, all_cases

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for case in all_cases():
        result = case.pupil.propgate(case.z_um, case.inputs, padding_factor=PADDING_FACTOR)
        np.savez(
            DATA_DIR / f"{case.name}.npz",
            field_x=result.field_x,
            field_y=result.field_y,
            field_z=result.field_z,
            x_um=result.x_um,
            y_um=result.y_um,
            intensity=result.intensity(normalize=False),
        )
        print(f"wrote data/{case.name}.npz")


if __name__ == "__main__":
    main()
