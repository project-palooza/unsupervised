import numpy as np
import pandas as pd

def windsorize(scaled):

    for col in scaled.columns:
        print(col)
        if np.max(scaled[col]) == 1.0:
            continue
        upper_limit = np.percentile(scaled[col],99)
        lower_limit = np.min(scaled[col])
        scaled[col] = np.clip(scaled[col],lower_limit,upper_limit)

    return scaled