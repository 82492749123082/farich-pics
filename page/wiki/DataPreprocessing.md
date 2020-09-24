## DataPreprocessing.py

### `class DataPreprocessing`

This class translates raw data into prepared datasets

#### `__init__(self, ellipse_format=True)`
Initialize class DataPreprocessing
> **Parameters**:
> * `ellipse_format`: set output format (`True` — ellipses, `False` — circles)

#### `get_axis_size(self, x_center, x_size, pmt_size, gap, chip_size, chip_num_size)`
Return pixels of the detector

> **Parameters**:
> * `x_center`: center point of the detector on x side (mm)
> * `x_size`: number of PMT on x side
> * `pmt_size`: size of one PMT (mm)
> * `gap`: distance between adjacent PMTs (mm)
> * `chip_size`: size of one pixel in PMT (mm)
> * `chip_num_size`: number of pixels in PMT on x side

> **Returns**: array from min to max coordinate of the detector with binning as pixels on one side

#### `get_board_size(self, info_arrays)`
Return pixels of the detector

> **Parameters**:
> * `info_arrays`: arrays from info_sim in ROOT-file

> **Returns**: `(xedges, yedges)` - x, y arrays

#### `parse_root(self, *rootFiles)`
Parse root files

> **Parameters**:
> * `rootFiles`: paths to ROOT-files

> **Returns**: `None`

#### `parse_pickle(self, *pickleFiles)`
Parse pickle files

> **Parameters**:
> * `pickleFiles`: paths to pickle files

> **Returns**: `None`

#### `process_root(self, *rootFiles)`
equal `parse_root` for `ellipse_format==True`

> **Parameters**:
> * `rootFiles`: paths to pickle files

#### `get_images(self)`
Return prepared dataset from previously parsed files

> **Parameters**: `None`

> **Returns**: `(X, y)` dataset. 
> `X` is the pixel detector, `y` is a vector of coordinate the circle center and its radius (in pixels)

#### `save_data(self, filename="data/temp.pkl")`
Save prepared dataset to pickle file

> **Parameters**:
> * `filename`: paths to output file

> **Returns**: `None`

#### `generate_boards(self, board_size, N_circles, N_boards)`
Generate (`N_boards`) boards with data from parsed files with `size=(board_size, board_size)`.
Numbers of circles on each board is equal to `N_circles`.

> **Parameters**:
> * `board_size`: axis size or generated boards (px)
> * `N_circles` : maximum number of circles
> * `N_boards` : number of generated boards

> **Returns**: `(H_all, h_all, mask_all)`
> * `H_all` : list of generated boards (as sparse arrays)
> * `h_all` : list of ellipses/circles parameters on a board
> * `mask_all` : list of ellipses/circles masks on a board

#### `generate_boards_randnum(self, board_size, N_circles, N_boards)`
Generate (`N_boards`) boards with data from parsed files with `size=(board_size, board_size)`.
Numbers of circles on each board between `[1, N_circles]`.

> **Parameters**:
> * `board_size`: axis size or generated boards (px)
> * `N_circles` : maximum number of circles
> * `N_boards` : number of generated boards

> **Returns**: `(H_all, h_all, mask_all)`
> * `H_all` : list of generated boards (as sparse arrays)
> * `h_all` : list of ellipses/circles parameters on a board
> * `mask_all` : list of ellipses/circles masks on a board

#### `generate_3d_boards(self, board_size, N_circles, N_boards)`
Generate 3D (`N_boards`) boards with data from parsed files with `size=(board_size, board_size)`.
Numbers of circles on each board is equal to `N_circles`.

> **Parameters**:
> * `board_size`: axis size or generated boards (px)
> * `N_circles` : maximum number of circles
> * `N_boards` : number of generated boards

> **Returns**: `H_all` - array of arrays with four columns `(x, y, time, ring_index)`

#### `generate_3d_boards_randnum(self, board_size, N_circles, N_boards)`
Generate (`N_boards`) boards with data from parsed files with `size=(board_size, board_size)`.
Numbers of circles on each board between `[1, N_circles]`.

> **Parameters**:
> * `board_size`: axis size or generated boards (px)
> * `N_circles` : maximum number of circles
> * `N_boards` : number of generated boards

> **Returns**: `H_all` - array of arrays with four columns `(x, y, time, ring_index)`