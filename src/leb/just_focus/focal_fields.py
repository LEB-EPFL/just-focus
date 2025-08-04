"""Electromagnetic fields in the focus of a high NA microscope objective."""
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dtypes import Complex, Float


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
