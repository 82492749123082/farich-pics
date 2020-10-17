import uproot
import numpy as np
import progressbar as pg
import warnings
from farichlib import Augmentator
import json


class BoardsGenerator:
    def __init__(self, *rootFiles):
        self.__events = None
        self.__boards = None
        self.__boards_sizes = None
        self.__freq = None
        self.__chip_size = None
        self.AddROOT(*rootFiles)

    def AddROOT(self, *rootFiles):
        for rootFile in rootFiles:
            info_arrays = uproot.open(rootFile)["info_sim"].arrays()
            raw_tree = uproot.open(rootFile)["raw_data"]
            xedges, yedges, chip_size = self.get_board_size(info_arrays)

            if self.__chip_size is None:
                self.__chip_size = chip_size
            elif self.__chip_size != chip_size:
                raise Exception("self.__chip_size should match previous value")

            df = raw_tree.pandas.df(
                branches=["hits.pos_chip._*", "hits", "hits.time", "id_event"],
            ).query("hits>10")
            df = df.rename(
                {
                    "hits.pos_chip._0": "chipx",
                    "hits.pos_chip._1": "chipy",
                    "hits.pos_chip._2": "chipz",
                    "hits.time": "time",
                    "id_event": "event",
                },
                axis=1,
            )
            df["chipx"] = np.digitize(df["chipx"], xedges)
            df["chipy"] = np.digitize(df["chipy"], yedges)
            grouped = df[["chipx", "chipy", "time", "event"]].groupby("entry")

            df_events = []
            for i, (name, group) in enumerate(pg.progressbar(grouped)):
                arr = group.values
                arr[:, 3] = i
                df_events.append(arr)
            events = np.concatenate(df_events)
            self.__write_data(events)
        return

    def __write_data(self, events):
        if self.__events is None:
            self.__events = events
        else:
            self.__events = np.vstack([self.__events, events])
        return

    def get_axis_size(self, x_center, x_size, pmt_size, gap, chip_size, chip_num_size):
        xmin = x_center - (x_size * pmt_size + (x_size - 1) * gap + chip_size) / 2
        xmax = x_center + (x_size * pmt_size + (x_size - 1) * gap + chip_size) / 2
        xbins = x_size * chip_num_size
        return np.linspace(xmin, xmax, xbins)

    def get_board_size(self, info_arrays):
        x_size = info_arrays[b"num_side_x"][0]
        x_center = info_arrays[b"origin_pos._0"][0]
        chip_size = info_arrays[b"chip_size"][0]
        chip_num_size = info_arrays[b"chip_num_size"][0]
        pmt_size = info_arrays[b"size"][0]
        gap = info_arrays[b"gap"][0]
        y_size = info_arrays[b"num_side_y"][0]
        y_center = info_arrays[b"origin_pos._1"][0]
        xedges = self.get_axis_size(
            x_center, x_size, pmt_size, gap, chip_size, chip_num_size
        )
        yedges = self.get_axis_size(
            y_center, y_size, pmt_size, gap, chip_size, chip_num_size
        )
        return (xedges, yedges, chip_size)

    def ClearROOT(self):
        self.__events = None

    def GenerateBoards(
        self,
        n_boards,
        n_rings=1,
        size=(100, 100),
        freq=300,
        ticks=200,
        noise_level=100,
        augmentations=[Augmentator.Shift],
    ):
        if isinstance(n_boards, int):
            pass
        else:
            raise TypeError("Incorrect type: n_boards must be int")
        if n_boards <= 0:
            raise ValueError("n_boards must be more than zero")

        if isinstance(n_rings, int):
            n_rings_min = n_rings_max = n_rings
        elif isinstance(n_rings, tuple):
            n_rings_min, n_rings_max = n_rings[0], n_rings[1]
        else:
            raise TypeError("Incorrect type: n_rings must be int or tuple")
        if n_rings_min < 0:
            raise ValueError("Bad minimum number of rings")
        if n_rings_min > n_rings_max:
            raise ValueError("Minimum number of rings more than maximum")

        if isinstance(size, tuple):
            pass
        else:
            raise TypeError("Incorrect type: size must be tuple")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Board_size less than/or zero")

        if self.__boards_sizes is None:
            self.__boards_sizes = (size[0], size[1], int(10 ** 9 / freq / ticks))
        elif self.__boards_sizes != (size[0], size[1], int(10 ** 9 / freq / ticks)):
            raise Exception("self.__boards_sizes should match previous values")

        if self.__freq is None:
            self.__freq = freq
        elif self.__freq != freq:
            raise Exception("self.__freq should match previous value")

        if self.__events is None:
            pass
        else:
            n_rings_rdm = np.random.randint(n_rings_min, n_rings_max + 1, n_boards)
            for i in pg.progressbar(range(0, n_boards)):
                # for i in range(0, n_boards):
                board = self.__generate_board(
                    n_rings=n_rings_rdm[i],
                    noise_level=noise_level,
                    augmentations=augmentations,
                )
                id_board_array = np.ones((boars.shape[0],1), int)*i                
                board = np.concatenate((board,id_board_array), axis=1)
                
                if self.__boards is None:
                    self.__boards = board
                else:
                    self.__boards = np.concatenate((self.__boards, board))
        return

    def __generate_board(self, n_rings, noise_level, augmentations):
        newboard = np.empty((0, 4), int)
        indices = np.random.randint(low=0, high=self.__events[-1, -1], size=n_rings)
        tedges = np.linspace(0, 1 / (self.__freq * 1000), self.__boards_sizes[2])
        
        for loc_ind in indices:
            loc_events = self.__events[self.__events[:, -1] == loc_ind]
            loc_events[:, 2] = np.digitize(loc_events[:, 2], tedges)

            loc_events[:, 0] -= np.median(loc_events[:, 0])
            loc_events[:, 1] -= np.median(loc_events[:, 1])
            loc_events[:, 2] -= np.median(loc_events[:, 2])
            loc_events[:, 3] = np.ones(loc_events.shape[0])
            
            newboard = self.__add_to_board(
                board=newboard,
                arr=loc_events.astype(int),
                augmentations=augmentations,
            )

        Augmentator.Noise(
            newboard,
            size=self.__boards_sizes,
            noise_level=noise_level,
            chip_size=self.__chip_size,
        )

        return newboard

    def __add_to_board(self, board, arr, augmentations):
        for aug in augmentations:
            aug(arr, size=self.__boards_sizes)
        mask = (
            (arr[:, 0] >= 0)
            & (arr[:, 0] < self.__boards_sizes[0])
            & (arr[:, 1] >= 0)
            & (arr[:, 1] < self.__boards_sizes[1])
            & (arr[:, 2] >= 0)
            & (arr[:, 2] < self.__boards_sizes[2])
        )
        arr = arr[mask]

        board = np.concatenate((board, arr))
        return board

    def SaveBoards(self, filepath="temp.json"):
        mydict = {"boards": self.__boards.tolist(), "sizes": self.__boards_sizes}
        with open(filepath, "w") as json_file:
            json.dump(mydict, json_file)
        return

    def GetBoards(self):
        return (self.__boards, self.__boards_sizes)


if __name__ == "__main__":
    boardsGen = BoardsGenerator(
        "../data/farichSimRes_mu-pi-_1500MeV_15-90deg_30.0k_2020-02-10.root"
    )
    boardsGen.GenerateBoards(100)
    print(boardsGen.GetBoards())
