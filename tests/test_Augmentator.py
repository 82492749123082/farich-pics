import pytest
import numpy as np
from farichlib import Augmentator


@pytest.mark.parametrize(
    "xShift, yShift, timeShift",
    [(0, 0, 0), (-10, -10, -10), (5, 5, 5), (32, -13, 4)],
)
def test_Shift(xShift, yShift, timeShift):
    event = np.zeros((4, 11), dtype=int)
    Augmentator.Shift(event, xShift, yShift, timeShift)
    assert event.shape == (4, 11)
    assert np.equal(event[0], xShift * np.ones(11))
    assert np.equal(event[1], yShift * np.ones(11))
    assert np.equal(event[2], timeShift * np.ones(11))
    assert np.equal(event[3], np.zeros(11))


@pytest.mark.parametrize(
    "rescaleFactor",
    [1, 2, 3, 0.5],
)
def test_Rescale(rescaleFactor):
    event = np.ones((4, 27), dtype=int)
    Augmentator.Rescale(event, rescaleFactor)
    assert event.shape == (4, 27)
    assert np.isclose(event[0], rescaleFactor * np.ones(27))
    assert np.isclose(event[1], rescaleFactor * np.ones(27))
    assert np.isclose(event[2], np.ones(27))
    assert np.isclose(event[3], np.ones(27))


@pytest.mark.parametrize(
    "rotateAngle, xCenter, yCenter",
    [(0, 0, 0), (180, 0, 0), (45, 1, 1)],
)
def test_Rotate(rotateAngle, xCenter, yCenter):
    one = np.ones(27)
    ang = np.deg2rad(rotateAngle * one)
    event = np.ones((4, 27), dtype=int)
    Augmentator.Rotate(event, rotateAngle, xCenter, yCenter)
    assert event.shape == (4, 27)
    assert np.isclose(
        event[0],
        (1 - xCenter) * np.cos(ang) + (1 - yCenter) * np.sin(ang),
    )
    assert np.isclose(
        event[1],
        -(1 - xCenter) * np.sin(ang) + (1 - yCenter) * np.cos(ang),
    )
    assert np.isclose(event[2], ones)
    assert np.isclose(event[3], ones)