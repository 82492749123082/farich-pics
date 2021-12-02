## Intro
We can process Raw MC files with farichlib.DataPreprocessing.
After that, we can just train some models.

## Examples

### How to process raw MC and save data

```python
from farichlib import DataPreprocessing
import pickle

dp = DataPreprocessing()
dp.parse_root('someMC1.root', 'someMC2.root', ...)
dp.save_data('data.pkl')
```

### How to get toy-boards with real rings from MC

```python
from farichlib import DataPreprocessing
import pickle

dp = DataPreprocessing()
dp.parse_pickle('data.pkl') #instead we could use dp.parse_root('someMC1.root')
H_all, h_all, mask_all = dp.generate_boards_randnum(board_size=100, N_circles=3, N_boards=100)
with open("../dataset/dataset_many_circles.pkl", "wb") as f:
    pickle.dump((H_all, h_all, mask_all), f)
```