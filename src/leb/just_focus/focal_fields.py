"""Electromagnetic fields in the focus of a high NA microscope objective."""
from __future__ import annotations

from dataclasses import dataclass

from .backend import be
from .dtypes import Array


@dataclass(frozen=True)
class FocalField:
    field_x: Array
    field_y: Array
    field_z: Array
    x_um: Array
    y_um: Array

    def intensity(self, normalize: bool = True) -> Array:
        I = be.abs(self.field_x)**2 + be.abs(self.field_y)**2 + be.abs(self.field_z)**2
        if normalize:
            return I / be.max(I)
        return I
