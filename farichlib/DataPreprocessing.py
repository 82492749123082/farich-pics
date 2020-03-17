import uproot
import pandas as pd
import numpy as np
import random
from scipy import sparse
import pickle
import matplotlib.pyplot as plt


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
        arr_size = arr.shape[0]
        # xc, yc - center of circle in coordinates of small square
        xc, yc = y[0], y[1]
        # x1, y1 - top left angle of square
        x1 = random.randint(-xc, board_size - 1 - xc)
        y1 = random.randint(-yc, board_size - 1 - yc)
        # print(x1, y1)
        for i in range(0, arr_size):
            for j in range(0, arr_size):
                if x1 + i in range(0, board_size) and y1 + j in range(0, board_size):
                    board[x1 + i][y1 + j] += arr[i][j]
        # print(x1+y[0], y1+y[1])
        y = np.array([y[1] + y1, y[0] + x1, y[2]])
        Y = np.concatenate((Y, y))
        return board, Y

    def generate_board(self, board_size, N_circles, random_seed=0, shuffle=False):
        newboard = np.zeros((board_size, board_size))
        Y_res = np.array([])

        max_index = N_circles if N_circles < self.y.shape[0] else self.y.shape[0]
        for loc_ind in range(0, max_index):
            if loc_ind % 200 == 0:
                print(loc_ind)
            H = self.X[loc_ind].toarray()
            h = self.y[loc_ind]
            newboard, Y_res = self.add_to_board(newboard, Y_res, H, h)
        Y_res = np.reshape(Y_res, (-1, 3))
        return newboard, Y_res


if __name__ == "__main__":
    DP = DataPreprocessing()
    DP.parse_root("../data/farichSimRes_pi-kaon-_1000MeV_0-90deg_50.0k_2020-02-11.root")
    print(DP.get_images())

def add_noise(board, noise_level=0.001):
    #some code
    return board
    
def print_board(H, h):
    xedges = np.linspace(0, H.shape[0], H.shape[0])
    yedges = np.linspace(0, H.shape[1], H.shape[1])
    
    fig = plt.figure(frameon=False, figsize=(50, 50) )
    ax = plt.Axes(fig, [0., 0., 1., (H.shape[1]/H.shape[0])])
    fig.add_axes(ax)
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H, cmap='gray')
    h = np.reshape(h, (-1,3))
    plt.scatter(h[:,0], h[:,1], marker='+', s=550, c='red') #mean vertex
