import uproot
import pandas as pd
import numpy as np
import random
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
    
    def add_to_board(self, board, Y, arr, y):
        board_size = board.shape[0]
        arr_size = arr.shape[0]
        #xc, yc - center of circle in coordinates of small square 
        xc, yc = y[0], y[1]
        #x1, y1 - top left angle of square
        x1 = random.randint(-xc, board_size-1-xc)
        y1 = random.randint(-yc, board_size-1-yc)
        #print(x1, y1)
        for i in range(0, arr_size):
            for j in range(0, arr_size):
                if x1+i in range(0,board_size) and y1+j in range(0,board_size):
                    board[x1+i][y1+j] += arr[i][j]  
        #print(x1+y[0], y1+y[1])
        y = np.array([y[1]+y1, y[0]+x1, y[2]])
        Y = np.concatenate((Y, y))
        return board, Y
    
    def generate_board(self, leng, N_circles, noise_level):
        newboard = np.zeros((leng, leng))
        Y_res = np.array([])
        
        max_index = N_circles if N_circles > self.y.shape[0] else self.y.shape[0]
        for loc_ind in range(0, max_index):
            if loc_ind % 200 == 0:
                print(loc_ind)
            H = self.X[loc_ind].toarray()
            arr = self.y[loc_ind]
            newboard, Y_res = self.add_to_board(newboard, Y_res, H, arr)
        Y_res = np.reshape(Y_res, (-1, 3))
        return newboard, Y_res
                                                                                            

if __name__ == "__main__":
    DP = DataPreprocessing('../data/farichSimRes_pi-kaon-_1000MeV_0-90deg_50.0k_2020-02-11.root')
    print(DP.get_images())
