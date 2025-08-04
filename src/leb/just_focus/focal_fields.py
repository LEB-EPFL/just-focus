"""Electromagnetic fields in the focus of a high NA microscope objective."""
from dataclasses import dataclass

import numpy as np


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
