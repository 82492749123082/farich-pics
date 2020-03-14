import uproot
import pandas as pd
import numpy as np
from scipy import sparse


class DataPreprocessing():

    def __init__(self, rootFile):
        info_tree = uproot.open(rootFile)['info_sim']
        raw_tree = uproot.open(rootFile)['raw_data']
        df = raw_tree.pandas.df(branches=['hits.pos_chip._0', 'hits.pos_chip._1', 'pos_primary._0', 'pos_primary._1'])
        df = df.rename({'hits.pos_chip._0': 'chip_x', 'hits.pos_chip._1': 'chip_y', 'pos_primary._0':'v_x', 'pos_primary._1':'v_y'}, axis=1)
        df['radius'] = np.sqrt((df['chip_x'] - df['v_x'])**2 + (df['chip_y'] - df['v_y'])**2)
        df = df.groupby('entry').agg({'chip_x': list, 'chip_y': list, 'v_x': 'mean', 'v_y': 'mean', 'radius': 'median'})

        # print(df.head())

        bins = np.linspace(-200, 200, 100)
        hists = []
        for i, j in df.iterrows():
            H, _, _ = np.histogram2d(j['chip_x'], j['chip_y'], bins=bins)
            hists.append(sparse.csr_matrix(H))

        hists = np.array(hists)
        zerobin = np.digitize( 0, bins)
        df['radius'] = np.digitize( df['radius'], bins) - zerobin
        df['v_x'] = np.digitize(df['v_x'], bins)
        df['v_y'] = np.digitize(df['v_y'], bins)

        self.X = hists
        self.y = df[['v_x', 'v_y', 'radius']].values

    def get_images(self):
        return (self.X, self.y)


if __name__ == "__main__":
    DP = DataPreprocessing('../data/farichSimRes_pi-kaon-_1000MeV_0-90deg_50.0k_2020-02-11.root')
    print(DP.get_images())
