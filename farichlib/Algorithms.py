import numpy as np


def classic_algo(board, threshold=10, cumulative=1):
    x, y, t = board.T[:3]
    cumulative = min(cumulative, len(t) - 2)
    tNum, bins = np.histogram(t, bins=np.arange(t.max()))
    tConv = np.convolve(tNum, np.ones(cumulative, dtype=int), "valid")
    cut = tConv > threshold
    min_bins, max_bins = bins[:-cumulative][cut], bins[cumulative:][cut]
    return np.any(
        (t.reshape((-1, 1)) >= min_bins) & (t.reshape((-1, 1)) < max_bins), axis=1
    )
