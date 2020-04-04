import pytest
import sys
from farichlib.Dataset import Dataset


def test_initial():
    ds = Dataset(0.1)
    assert ds.imgs is None
    assert ds.masks is None
    assert ds.circles is None
    assert ds.noise == 0.1


def test_load():
    ds = Dataset(0.01)
    ds.load("tests/Boards_example_data.pkl")
    assert len(ds.imgs) == len(ds.masks)
    assert len(ds.imgs) == len(ds.circles)


def test_item():
    ds = Dataset(0.01)
    ds.load("tests/Boards_example_data.pkl")
    result = ds[0]
    assert result[0] is not None
    assert "boxes" in result[1]
    assert "labels" in result[1]
    assert "masks" in result[1]
    assert "image_id" in result[1]
    assert "area" in result[1]


def test_len():
    ds = Dataset(0.01)
    ds.load("tests/Boards_example_data.pkl")
    assert len(ds) == 10
