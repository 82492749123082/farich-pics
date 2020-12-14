import pytest
import numpy as np
from farichlib import Algorithms


def test_classic_algo():
    times = np.array([9, 1, 1, 1, 0, 1, 1, 1, 2, 1], dtype=int)
    freqs = np.array(
        [1, 7, 7, 7, 1, 7, 7, 7, 1, 7], dtype=int
    )  # 9 встречается 1 раз, 1 - 7 раз и т.д.
    data = np.concatenate(
        [
            np.random.randint(0, 100, (10, 1)),
            np.random.randint(0, 100, (10, 1)),
            times.reshape((-1, 1)),
        ],
        axis=1,
    )
    y_pred = Algorithms.classic_algo(data, threshold=6, cumulative=1)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (10,)
    assert np.all(y_pred == (times == 1))
    y_pred = Algorithms.classic_algo(data, threshold=6, cumulative=2)
    assert np.all(y_pred == (times <= 2))
    y_pred = Algorithms.classic_algo(data, threshold=11, cumulative=11)
    assert ~np.any(y_pred)
    # Test threshold=None
    counts = Algorithms.classic_algo(data, threshold=None)
    assert isinstance(counts, np.ndarray)
    assert np.any(np.isclose(counts, freqs))