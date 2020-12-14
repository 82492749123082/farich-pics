import numpy as np


def __handle_errors_classic_algo(board, threshold, cumulative):
    if threshold is not None:
        if not (isinstance(threshold, int) or isinstance(threshold, float)):
            raise TypeError(
                f"Argument threshold must be float or int or None, not {type(threshold)}"
            )
        if threshold < 0:
            raise ValueError("Threshold less zero")
    if not (isinstance(cumulative, int)):
        raise TypeError(f"Argument cumulative must be int, not {type(cumulative)}")
    if cumulative < 1:
        raise ValueError("Cumulative less one")


def classic_algo(board, threshold=10, cumulative=1):
    """
    Классический алгоритм поиска сигнальных фотонов
    """
    __handle_errors_classic_algo(board, threshold, cumulative)
    x, y, t = board.T[:3]
    cumulative = min(cumulative, len(t) - 2)
    hist_bins = np.arange(t.max() + 2)
    tNum, bins = np.histogram(t, bins=hist_bins)
    tConv = np.convolve(tNum, np.ones(cumulative, dtype=int), "valid")
    if threshold is None:
        filled_bins = np.append(
            tConv,
            np.zeros(cumulative - 1),
        )
        count_photons = np.max(
            [np.roll(filled_bins, i) for i in range(cumulative)], axis=0
        )
        return count_photons[t]
    cut = tConv > threshold
    min_bins, max_bins = bins[:-cumulative][cut], bins[cumulative:][cut]
    events = np.any(
        (t.reshape((-1, 1)) >= min_bins) & (t.reshape((-1, 1)) < max_bins), axis=1
    )
    return events