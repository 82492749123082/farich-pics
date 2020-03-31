import pytest
import sys
from farichlib.DataPreprocessing import DataPreprocessing


def test_default_parameters():
    dp = DataPreprocessing()
    assert dp.X is None
    assert dp.y is None


def test_axis_size():
    dp = DataPreprocessing()
    edges = dp.get_axis_size(1, 3, 2, 3, 0.1, 10)
    assert len(edges) == 30


def test_parsing():
    dp = DataPreprocessing()
    dp.parse_pickle("tests/DataPreprocessing_example_data.pkl")
    assert len(dp.X) == 10
    assert len(dp.y) == 10


def test_board_generation():
    dp = DataPreprocessing()
    dp.parse_pickle("tests/DataPreprocessing_example_data.pkl")
    imgs, boxes, masks = dp.generate_boards(100, 1, 10)
    assert len(imgs) == len(masks)
    assert len(boxes) == len(imgs)
    assert len(boxes) == 10
    assert imgs[0].shape == (100, 100)
    assert len(boxes[0]) == 1
