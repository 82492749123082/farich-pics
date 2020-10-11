import pytest
from farichlib import BoardsGenerator


@pytest.mark.parametrize(
    "n_boards, sizeX, sizeY",
    [(0, 100, 432), (10, 14, 12), (13, 93, 13), (88, 144, 144)],
)
def test_complex(n_boards, sizeX, sizeY):
    bg = BoardsGenerator()
    bg.GenerateBoards(n_boards, n_rings=(1, 1), size=(sizeX, sizeY), noise_level=1e3)
    boards, sizes = bg.GetBoards()
    print(boards)
    assert sizes[0] == sizeX
    assert sizes[1] == sizeY
    if n_boards == 0:
        assert boards.size == 0
    else:
        assert boards.shape[0] == 5
        assert (boards[4, :]).max() == n_boards - 1
