import numpy as np
import pandas as pd

feature_list_mini = []
df = pd.DataFrame(
    [[5, 5, 5, 5],
     [6, 6, 6, 6],
     [7, 7, 7, 7],
     [8, 8, 8, 8],
     [1, 1, 1, 1]]
)

# 第三四分位
feature_list_mini.extend(np.percentile(df.values, 75, axis=0)-np.percentile(df.values, 25, axis=0))

print(feature_list_mini)