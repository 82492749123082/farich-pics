import pytest
import numpy as np
from farichlib import BoardsGenerator


@pytest.mark.parametrize(
    "n_boards, sizeX, sizeY",
    [(1, 10, 12), (10, 74, 72), (13, 93, 63), (25, 144, 144)],
)
def test_noise_generator(n_boards, sizeX, sizeY):
    bg = BoardsGenerator("tests/testTree.root")
    bg.GenerateBoards(n_boards, n_rings=(1, 1), size=(sizeX, sizeY), noise_level=1e2)
    boards, sizes = bg.GetBoards()
    assert sizes[0] == sizeX
    assert sizes[1] == sizeY
    assert boards.shape[1] == 5
    bd = bg.GetBoards()
    assert (boards[:, 4]).max() == n_boards - 1


@pytest.mark.parametrize(
    "n_boards, sizeX, sizeY",
    [(1, 10, 12), (10, 74, 72), (13, 93, 63), (25, 144, 144)],
)
def test_empty(n_boards, sizeX, sizeY):
    bg = BoardsGenerator()
    bg.GenerateBoards(n_boards, size=(sizeX, sizeY), noise_level=1e2)
    boards, sizes = bg.GetBoards()
    assert sizes[0] == sizeX
    assert sizes[1] == sizeY
    assert boards.shape[1] == 5
    bd = bg.GetBoards()
    assert (boards[:, 4]).max() == n_boards - 1


def test_ClearROOT_and_UniformNoise():
    bg = BoardsGenerator("tests/testTree.root")
    bg.ClearROOT()
    bg.GenerateBoards(10, (1, 20))
    boards, sizes = bg.GetBoards()
    assert np.all(boards[:, 3] == 0)  # check all pixels are noise
    for i in range(3):
        assert (
            sizes[i] * (0.5 - 2 / np.sqrt(12))
            < np.mean(boards[:, i])
            < sizes[i] * (0.5 + 2 / np.sqrt(12))
        )  # check noise uniform distribution (mean +/- 2*sigma)


def test_UniformCircles():
    bg = BoardsGenerator("tests/testTree.root")
    bg.GenerateBoards(1, 100, size=(100, 100), noise_level=0)
    boards, sizes = bg.GetBoards()
    assert np.all(boards[:, 3] == 1)
    for i in range(3):
        assert (
            sizes[i] * (0.5 - 2 / np.sqrt(12))
            < np.mean(boards[:, i])
            < sizes[i] * (0.5 + 2 / np.sqrt(12))
        )  # check signal circles uniform distribution (mean +/- 2*sigma)
