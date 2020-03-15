import uproot
import pandas as pd
import numpy as np
from scipy import sparse
import pickle


class DataPreprocessing:
    def __get_axis_size(
        self, x_center, x_size, pmt_size, gap, chip_size, chip_num_size
    ):
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

        xedges = self.__get_axis_size(
            x_center, x_size, pmt_size, gap, chip_size, chip_num_size
        )
        yedges = self.__get_axis_size(
            y_center, y_size, pmt_size, gap, chip_size, chip_num_size
        )

        return (xedges, yedges)

    def __init__(self):
        self.X = None
        self.y = None
        self.df = None

    def parse_root(self, rootFile):
        info_arrays = uproot.open(rootFile)["info_sim"].arrays()
        raw_tree = uproot.open(rootFile)["raw_data"]

        xedges, yedges = self.__get_board_size(info_arrays)
        zerobin = np.digitize(0, xedges)
        df = raw_tree.pandas.df(
            branches=[
                "hits.pos_chip._0",
                "hits.pos_chip._1",
                "pos_primary._0",
                "pos_primary._1",
            ]
        )
        df = df.rename(
            {
                "hits.pos_chip._0": "chipx",
                "hits.pos_chip._1": "chipy",
                "pos_primary._0": "px",
                "pos_primary._1": "py",
            },
            axis=1,
        )
        df["radius"] = np.sqrt(
            (df["chipx"] - df["px"]) ** 2 + (df["chipy"] - df["py"]) ** 2
        )
        df["chipx"] = np.digitize(df["chipx"], xedges)
        df["chipy"] = np.digitize(df["chipy"], yedges)
        df["data"] = np.ones(len(df))
        df["radius"] = np.digitize(df["radius"], xedges) - zerobin
        df["px"] = np.digitize(df["px"], xedges)
        df["py"] = np.digitize(df["py"], yedges)

        self.X = (
            df.groupby("entry")
            .apply(
                lambda x: sparse.coo_matrix(
                    (x["data"], (x["chipx"], x["chipy"])),
                    shape=(len(xedges), len(yedges)),
                )
            )
            .values
        )
        self.y = (
            df[["px", "py", "radius"]]
            .groupby("entry")
            .agg({"px": "mean", "py": "mean", "radius": "median"})
            .values
        )
        self.df = df

    def get_images(self):
        return (self.X, self.y)

    def save_data(self, filename="data/temp.pkl"):
        with open(filename, "wb") as f:
            pickle.dump((self.X, self.y), f)
        return

    def parse_pickle(self, pickleFile):
        with open(pickleFile, "rb") as f:
            self.X, self.y = pickle.load(f)
        return


if __name__ == "__main__":
    DP = DataPreprocessing()
    DP.parse_root("../data/farichSimRes_pi-kaon-_1000MeV_0-90deg_50.0k_2020-02-11.root")
    print(DP.get_images())
