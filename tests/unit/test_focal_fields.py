import numpy as np

from leb.just_focus import Float, FocalField


def test_focal_field_intensity_return_type_is_float():
    field_x = np.random.uniform(-1, 1, size=(4, 4)) + 1j * np.random.uniform(-1, 1, size=(4, 4))
    field_y, field_z = field_x.copy(), field_x.copy()
    x_um, y_um = np.meshgrid([0, 1, 2, 3], [0, 1, 2, 3])

    focal_field = FocalField(field_x, field_y, field_z, x_um, y_um)
    intensity = focal_field.intensity()

    assert intensity.dtype == Float
