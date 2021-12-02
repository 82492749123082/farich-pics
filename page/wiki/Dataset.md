## Dataset.py

### `class Dataset`
This class translates processed data to Mask-RCNN desirable view

#### `__init__(self, noise_level=(0, 0))`
Initialize class Dataset and define noise level which will be added to loaded in this class data
> **Parameters**:
> * `noise_level`: set (minimum, maximum) noise level.
> Noise level will be randomly selected from this range for each board.

#### `load(self, file)`
Load processed data (toy boards)
> **Parameters**:
> * `file`: path to processed data

#### `__getitem__(self, index)`
Return desirable for Mask-RCNN element with noise
> **Parameters**:
> * `index`: element index

#### `__len__(self)`
Return number of loaded boards
> **Returns**:
> * `len`: number of loaded boards