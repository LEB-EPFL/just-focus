import numpy as np

from leb.just_focus import Pupil


def test_pad_width():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    padding_factor = 2
    expected_padding = (3, 3)  # Add 3 elements to each side to get 2 * 2**2 = 8

    result = Pupil._pad_width(arr.shape, padding_factor=padding_factor)

    assert result[0] == expected_padding
    assert result[1] == expected_padding
