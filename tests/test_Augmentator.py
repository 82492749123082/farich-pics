import pytest
import numpy as np
from farichlib import Augmentator


def equality_test(data1, data2):
    return np.all(np.isclose(data1, data2))


@pytest.mark.parametrize(
    "xShift, yShift, timeShift",
    [(0, 0, 0), (-10, -10, -10), (5, 5, 5), (32, -13, 4)],
)
def test_Shift(xShift, yShift, timeShift):
    size = (11, 4)
    event = np.zeros(size, dtype=int)
    Augmentator.Shift(event, (100, 100, 100), xShift, yShift, timeShift)
    assert event.shape == size
    assert equality_test(event[:, 0], xShift * np.ones(11))
    assert equality_test(event[:, 1], yShift * np.ones(11))
    assert equality_test(event[:, 2], timeShift * np.ones(11))
    assert equality_test(event[:, 3], np.zeros(11))


@pytest.mark.parametrize(
    "rescaleFactor",
    [1, 2, 3, 0.5],
)
def test_Rescale(rescaleFactor):
    size = (27, 4)
    one = 2 * np.ones(27)
    event = 2 * np.ones(size, dtype=int)
    Augmentator.Rescale(event, (100, 100, 100), rescaleFactor)
    assert event.shape == size
    assert equality_test(event[:, 0], rescaleFactor * one)
    assert equality_test(event[:, 1], rescaleFactor * one)
    assert equality_test(event[:, 2], one)
    assert equality_test(event[:, 3], one)


@pytest.mark.parametrize(
    "rotateAngle, xCenter, yCenter",
    [(0, 2, 2), (180, 2, 2), (45, 1, 1), (90, 2, 2), (270, 2, 2)],
)
def test_Rotate(rotateAngle, xCenter, yCenter):
    size = (27, 4)
    one = np.ones(27)
    ang = np.deg2rad(rotateAngle * one)
    event = np.ones(size, dtype=int)
    Augmentator.Rotate(event, (100, 100, 100), rotateAngle, xCenter, yCenter)
    assert event.shape == size
    if rotateAngle == 0:
        assert equality_test(event[:, 0], one)
        assert equality_test(event[:, 1], one)
    if rotateAngle == 180:
        assert equality_test(event[:, 0], 3 * one)
        assert equality_test(event[:, 1], 3 * one)
    if rotateAngle == 45:
        assert equality_test(event[:, 0], one)
        assert equality_test(event[:, 1], one)
    if rotateAngle == 90:
        assert equality_test(event[:, 0], 3 * one)
        assert equality_test(event[:, 1], one)
    if rotateAngle == 270:
        assert equality_test(event[:, 0], one)
        assert equality_test(event[:, 1], 3 * one)
    assert equality_test(event[:, 2], one)
    assert equality_test(event[:, 3], one)