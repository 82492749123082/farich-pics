import pytest
import sys

sys.path.append(".")
from farichlib.DataPreprocessing import DataPreprocessing


def test_default_parameters():
    dp = DataPreprocessing()
    assert dp.X is None
    assert dp.y is None


def test_axis_size():
    dp = DataPreprocessing()
    edges = dp.get_axis_size(1, 3, 2, 3, 0.1, 10)
    assert len(edges) == 30
