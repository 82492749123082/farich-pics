import uproot
import pandas as pd
import numpy as np
import random
from scipy import sparse
import pickle
import matplotlib.pyplot as plt
import cv2
import progressbar as pg
import warnings


class DataPreprocessing:
    def __init__(self, ellipse_format=True):
        self.X = None
        self.y = None
        self.ellipse_format = ellipse_format
        if ellipse_format is False:
            warnings.warn(
                "This option will be deprecated in near future", DeprecationWarning
            )

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

        return (xedges, yedges)

    def __write_data(self, X, y):
        if (self.X is None) or (self.y is None):
            self.X = X
            self.y = y
        else:
            self.X = np.append(self.X, X)
            self.y = np.append(self.y, y, axis=0)
        return

    def __process_root(self, *rootFiles):
        print("ROOT handling")
        for rootFile in rootFiles:
            info_arrays = uproot.open(rootFile)["info_sim"].arrays()
            raw_tree = uproot.open(rootFile)["raw_data"]
            xedges, yedges = self.get_board_size(info_arrays)
            df = raw_tree.pandas.df(
                branches=["hits.pos_chip._*", "hits", "hits.time"], entrystop=10,
            ).query("hits>10")
            df = df.rename(
                {
                    "hits.pos_chip._0": "chipx",
                    "hits.pos_chip._1": "chipy",
                    "hits.pos_chip._2": "chipz",
                    "hits.time": "time",
                },
                axis=1,
            )
            df["chipx"] = np.digitize(df["chipx"], xedges)
            df["chipy"] = np.digitize(df["chipy"], yedges)
            df["data"] = np.ones(len(df))

            xmin, xmax = df["chipx"].value_counts()[:2].index
            ymin, ymax = df["chipy"].value_counts()[:2].index
            params = np.array(
                [
                    (xmax + xmin) / 2,
                    (ymax + ymin) / 2,
                    abs(xmax - xmin),
                    abs(ymax - ymin),
                    0,
                ]
            )
            grouped = df[["chipx", "chipy", "time", "data"]].groupby("entry")
            X = np.array([group.values for name, group in pg.progressbar(grouped)])
            y = np.broadcast_to(params, (len(X), 5))
            self.__write_data(X, y)
        return

    def parse_root(self, *rootFiles):
        if self.ellipse_format is True:
            self.__process_root(*rootFiles)
            return
        for rootFile in rootFiles:
            info_arrays = uproot.open(rootFile)["info_sim"].arrays()
            raw_tree = uproot.open(rootFile)["raw_data"]

            xedges, yedges = self.get_board_size(info_arrays)
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
            y = np.hstack((y, y[:, 2:3], np.zeros((y.shape[0], 1))))
            self.__write_data(X, y)
        return

    def parse_pickle(self, *pickleFiles):
        for pickleFile in pickleFiles:
            with open(pickleFile, "rb") as f:
                X, y = pickle.load(f)
            if y.shape[1] != 5:
                raise Exception("Old pickle, parse root again.")
            self.ellipse_format = True
            self.__write_data(X, y)
        return

    def get_images(self):
        return (self.X, self.y)

    def add_to_board(self, board, Y, arr, y):
        if board.shape[0] != board.shape[1]:
            raise ValueError("Incorrect board sizes")
        board_size = board.shape[0]
        xc, yc = y[0], y[1]

        arr_ = arr.T[[0, 1, 3]]  # x, y, data
        arr_[0], xc = arr_[0] - xc, 0
        arr_[1], yc = arr_[1] - yc, 0
        Augmentator.rotate(arr_, y)
        Augmentator.rescale(arr_, y)

        x1, y1 = np.random.randint(0, board.shape[0], 2)
        arr_[0], arr_[1] = arr_[0] + x1, arr_[1] + y1
        crop_ind = (
            (arr_[0] >= 0)
            & (arr_[0] < board_size)
            & (arr_[1] >= 0)
            & (arr_[1] < board_size)
        )
        arr_ = arr_[:, crop_ind].astype(int)
        xcoord, ycoord, data = arr_

        board.data = np.concatenate((board.data, data))
        board.row = np.concatenate((board.row, xcoord))
        board.col = np.concatenate((board.col, ycoord))

        yn = np.array([xc + x1, yc + y1, y[2], y[3], y[4]])
        Y = np.concatenate((Y, yn))
        return board, Y

    def generate_board(self, board_size, N_circles, random_seed=0, shuffle=False):
        newboard = sparse.coo_matrix(
            (np.array([]), (np.array([]), np.array([]))), shape=(board_size, board_size)
        )
        Y_res = np.array([])

        indices = np.random.randint(low=0, high=self.y.shape[0], size=N_circles)
        for loc_ind in indices:
            H = self.X[loc_ind].copy()
            h = self.y[loc_ind].copy()
            newboard, Y_res = self.add_to_board(newboard, Y_res, H, h)
        Y_res = np.reshape(Y_res, (-1, 5))
        return newboard, Y_res

    def __generate_boards(self, board_size, N_circles_min, N_circles_max, N_boards):
        if board_size < 0:
            raise ValueError("Board_size less than zero")
        if N_circles_min < 0:
            raise ValueError("Bad minimum number of circles")
        if N_circles_min > N_circles_max:
            raise ValueError("Minimum number of circles more than maximum")
        if N_boards <= 0:
            raise ValueError("N_boards must be more than zero")
        H_all, h_all, mask_all = [], [], []
        N_circles_rdm = np.random.randint(N_circles_min, N_circles_max + 1, N_boards)
        for i in pg.progressbar(range(0, N_boards)):
            board, Y_res = self.generate_board(
                board_size=board_size, N_circles=N_circles_rdm[i]
            )
            H_all.append(board)
            h_all.append(Y_res)
        mask_all = self.__create_masks(board_size, h_all)
        return H_all, h_all, mask_all

    def __create_masks(self, size, y_all):
        """
        create masks for all ellipses on each board
        """
        masks = list()
        for y in y_all:
            masks_one = list()
            for ellipse in y:
                mask = np.zeros((size, size), dtype=np.int8)
                e = ellipse.astype(int)
                cv2.ellipse(
                    mask,
                    (e[0], e[1]),
                    (e[2] // 2, e[3] // 2),
                    ellipse[4],
                    0,
                    360,
                    1,
                    -1,
                )
                masks_one.append(mask.astype(bool).T)
            masks.append(masks_one)
        return masks

    def generate_boards(self, board_size, N_circles, N_boards):
        """
        Generate `N_boards` boards with number rings equal `N_circles` per board
        and with size equal `(board_size, board_size)`
        Return `(H_all, h_all, mask_all)` - images, ellipses parameters, masks
        """
        return self.__generate_boards(board_size, N_circles, N_circles, N_boards)

    def generate_boards_randnum(self, board_size, N_circles, N_boards):
        """
        Generate `N_boards` boards with number rings from `1` to `N_circles` per board
        and with size equal `(board_size, board_size)`.
        Return `(H_all, h_all, mask_all)` - images, ellipses parameters, masks
        """
        return self.__generate_boards(board_size, 1, N_circles, N_boards)

    def save_boards(self, H_all, h_all, mask_all, filename):
        """
        save toy-boards to file (`filename`)
        """
        with open(filename, "wb") as f:
            pickle.dump((H_all, h_all, mask_all), f)
        return

    def save_data(self, filename="data/temp.pkl"):
        """
        save parsed ROOT-files to lightweight pickle
        """
        with open(filename, "wb") as f:
            pickle.dump((self.X, self.y), f)
        return


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


class Augmentator:
    """
    static class as a set of methods to augment data
    """

    def rotate(H, y, angle=None):
        """
        rotation around `(0,0)`
        """
        if len(y) == 3:
            return H
        if angle is None:
            angle = random.random() * 180
        xc, yc = 0, 0
        y[-1] += angle
        xcoord, ycoord = H[[0, 1]]
        cos, sin = np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)
        M = np.array([[cos, -sin], [sin, cos]])
        rot = (M @ np.array([xcoord - xc, ycoord - yc])).astype(int)
        H[[0, 1]] = rot[0] + xc, rot[1] + yc
        return

    def rescale(H, y, zoom=None):  # rescale
        if zoom is None:
            zoom = 0.5 + random.random()
        xc, yc = 0, 0
        y[2], y[3] = zoom * y[2], zoom * y[3]
        rescaling = (zoom * np.array([H[0] - xc, H[1] - yc])).astype(int)
        H[[0, 1]] = rescaling[0] + xc, rescaling[1] + yc
        return


if __name__ == "__main__":
    DP = DataPreprocessing()
    DP.parse_root("../data/farichSimRes_pi-kaon-_1000MeV_0-90deg_50.0k_2020-02-11.root")
    print(DP.get_images())
