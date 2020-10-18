import pytest
from farichlib import BoardsGenerator


@pytest.mark.parametrize(
    "n_boards, sizeX, sizeY",
    [(1, 10, 12), (10, 74, 72), (13, 93, 63), (88, 144, 144)],
)
def test_noise_generator(n_boards, sizeX, sizeY):
    bg = BoardsGenerator("tests/testTree.root")
    bg.GenerateBoards(n_boards, n_rings=(1, 1), size=(sizeX, sizeY), noise_level=1e3)
    boards, sizes = bg.GetBoards()
    assert sizes[0] == sizeX
    assert sizes[1] == sizeY
    assert boards.shape[1] == 5
    bd = bg.GetBoards()
    print(bd)
    assert (boards[:, 4]).max() == n_boards - 1

@pytest.mark.parametrize(
    "n_boards, sizeX, sizeY",
    [(1, 10, 12), (10, 74, 72), (13, 93, 63), (88, 144, 144)],
)    
def test_empty(n_boards, sizeX, sizeY):
    bg = BoardsGenerator()
    bg.GenerateBoards(n_boards, size=(sizeX, sizeY), noise_level=1e3)
    boards, sizes = bg.GetBoards()
    assert sizes[0] == sizeX
    assert sizes[1] == sizeY
    assert boards.shape[1] == 5
    bd = bg.GetBoards()
    print(bd)
    assert (boards[:, 4]).max() == n_boards - 1


