import uproot
import pandas as pd
import numpy as np
import random
from scipy import sparse
import pickle
import matplotlib.pyplot as plt
from numba import jit, njit
from numba.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaWarning,
)
import warnings

# Suppress numba warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)


class DataPreprocessing:
    def get_axis_size(self, x_center, x_size, pmt_size, gap, chip_size, chip_num_size):
        xmin = x_center - (x_size * pmt_size + (x_size - 1) * gap + chip_size) / 2
        xmax = x_center + (x_size * pmt_size + (x_size - 1) * gap + chip_size) / 2
        xbins = x_size * chip_num_size
        return np.linspace(xmin, xmax, xbins)

    def __get_board_size(self, info_arrays):
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

        return (xedges, yedges)

    def __write_data(self, X, y):
        if (self.X is None) or (self.y is None):
            self.X = X
            self.y = y
        else:
            self.X = np.append(self.X, X)
            self.y = np.append(self.y, y, axis=0)
        return

    def __init__(self):
        self.X = None
        self.y = None
        self.df = None

    def parse_root(self, *rootFiles):
        for rootFile in rootFiles:
            info_arrays = uproot.open(rootFile)["info_sim"].arrays()
            raw_tree = uproot.open(rootFile)["raw_data"]

            xedges, yedges = self.__get_board_size(info_arrays)
            zerobin = np.digitize(0, xedges)
            df = raw_tree.pandas.df(
                branches=["hits.pos_chip._*", "pos_primary._*", "dir_primary._*"]
            )
            df = df.rename(
                {
                    "hits.pos_chip._0": "chipx",
                    "hits.pos_chip._1": "chipy",
                    "hits.pos_chip._2": "chipz",
                    "pos_primary._0": "px",
                    "pos_primary._1": "py",
                    "pos_primary._2": "pz",
                    "dir_primary._0": "vx",
                    "dir_primary._1": "vy",
                    "dir_primary._2": "vz",
                },
                axis=1,
            )

            df["px"] += (df["chipz"] - df["pz"]) * df["vx"]
            df["py"] += (df["chipz"] - df["pz"]) * df["vy"]

            df["radius"] = np.sqrt(
                (df["chipx"] - df["px"]) ** 2 + (df["chipy"] - df["py"]) ** 2
            )
            df["chipx"] = np.digitize(df["chipx"], xedges)
            df["chipy"] = np.digitize(df["chipy"], yedges)
            df["data"] = np.ones(len(df))
            df["radius"] = np.digitize(df["radius"], xedges) - zerobin
            df["px"] = np.digitize(df["px"], xedges)
            df["py"] = np.digitize(df["py"], yedges)

            X = (
                df.groupby("entry")
                .apply(
                    lambda x: sparse.coo_matrix(
                        (x["data"], (x["chipx"], x["chipy"])),
                        shape=(len(xedges), len(yedges)),
                    )
                )
                .values
            )
            y = (
                df[["px", "py", "radius"]]
                .groupby("entry")
                .agg({"px": "mean", "py": "mean", "radius": "median"})
                .values
            )

            self.__write_data(X, y)
        return

    def parse_pickle(self, *pickleFiles):
        for pickleFile in pickleFiles:
            with open(pickleFile, "rb") as f:
                X, y = pickle.load(f)
            self.__write_data(X, y)
        return

    def save_data(self, filename="data/temp.pkl"):
        with open(filename, "wb") as f:
            pickle.dump((self.X, self.y), f)
        return

    def get_images(self):
        return (self.X, self.y)

    def add_to_board(self, board, Y, arr, y):
        board_size = board.shape[0]
        # cropping self.X arrays to get better result
        xc = y[0]
        yc = y[1]
        r = y[2]
        xlow = int(xc - 1.5 * r) if (xc - 1.5 * r > 0) else 0
        xhigh = (
            int(xc + 1.5 * r) if (xc + 1.5 * r < arr.shape[0]) else (arr.shape[0] - 1)
        )
        ylow = int(yc - 1.5 * r) if (yc - 1.5 * r > 0) else 0
        yhigh = (
            int(yc + 1.5 * r) if (yc + 1.5 * r < arr.shape[1]) else (arr.shape[1] - 1)
        )
        arr_ = arr.toarray()[xlow:xhigh, ylow:yhigh]
        arr = sparse.coo_matrix(arr_)
        xc = xc - xlow
        yc = yc - ylow

        x1, y1 = np.random.randint(0, board.shape[0] - arr.shape[0], 2)

        board.data = np.concatenate((board.data, arr.data))
        board.row = np.concatenate((board.row, arr.row + x1))
        board.col = np.concatenate((board.col, arr.col + y1))

        y = np.array([yc + y1, xc + x1, r])
        Y = np.concatenate((Y, y))
        return board, Y

    def generate_board(self, board_size, N_circles, random_seed=0, shuffle=False):
        newboard = sparse.coo_matrix(
            (np.array([]), (np.array([]), np.array([]))), shape=(board_size, board_size)
        )
        Y_res = np.array([])

        indices = np.random.randint(low=0, high=self.y.shape[0], size=N_circles)
        for loc_ind in indices:
            H = self.X[loc_ind]
            h = self.y[loc_ind]
            newboard, Y_res = self.add_to_board(newboard, Y_res, H, h)
        Y_res = np.reshape(Y_res, (-1, 3))
        return newboard, Y_res

    @jit(nopython=False)
    def generate_boards(self, board_size, N_circles, N_boards):
        H_all = []
        h_all = []
        mask_all = []
        for i in range(0, N_boards):
            # if i % 5000 == 0:
            #     print(i)
            board, Y_res = self.generate_board(
                board_size=board_size, N_circles=N_circles
            )
            H_all.append(board)
            h_all.append(Y_res)
            mask_all.append(create_mask(board_size=board_size, Y_res=Y_res))
        return H_all, h_all, mask_all

    @jit(nopython=False)
    def generate_boards_randnum(self, board_size, N_circles, N_boards):
        H_all = []
        h_all = []
        mask_all = []
        for i in range(0, N_boards):
            if i % 5000 == 0:
                print(i)
            N_circles_rdm = random.randint(1, N_circles)
            board, Y_res = self.generate_board(
                board_size=board_size, N_circles=N_circles_rdm
            )
            H_all.append(board)
            h_all.append(Y_res)
            mask_all.append(create_mask(board_size=board_size, Y_res=Y_res))
        return H_all, h_all, mask_all


if __name__ == "__main__":
    DP = DataPreprocessing()
    DP.parse_root("../data/farichSimRes_pi-kaon-_1000MeV_0-90deg_50.0k_2020-02-11.root")
    print(DP.get_images())


@njit
def create_mask_addit(board_size, Y_res):
    # now only for circles
    x = np.linspace(0, board_size, board_size)
    y = np.linspace(0, board_size, board_size).reshape((-1, 1))
    mask_joined = []
    for index in range(Y_res.shape[0]):
        x0 = Y_res[index][0]
        y0 = Y_res[index][1]
        R = Y_res[index][2]
        circle = np.nonzero((x - x0) ** 2 + (y - y0) ** 2 <= R ** 2)
        mask_joined.append(circle)
    return mask_joined


def create_mask(board_size, Y_res):
    masks = create_mask_addit(board_size, Y_res)
    return list(
        map(
            lambda x: sparse.csc_matrix(
                (np.ones(len(x[0])), x), shape=(board_size, board_size)
            ),
            masks,
        )
    )


def print_board(H, h):
    H = H.toarray()
    xedges = np.linspace(0, H.shape[0], H.shape[0])
    yedges = np.linspace(0, H.shape[1], H.shape[1])

    #    fig = plt.figure(frameon=False, figsize=(50, 50))
    fig = plt.figure(frameon=False, figsize=(5, 5))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, (H.shape[1] / H.shape[0])])
    fig.add_axes(ax)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H, cmap="gray")
    h = np.reshape(h, (-1, 3))
    plt.scatter(h[:, 0], h[:, 1], marker="+", s=550, c="red")  # mean vertex
    return
