# code that scales the data
import pandas as pd
from sklearn.preprocessing import RobustScaler


def robust_scale(df):
# instantiate the thing (in this case the robustscaler)
    scaler = RobustScaler()

    # fit the thing (our scaler)
    scaler.fit(df)

    # transform the data using the thing
    scaled = scaler.transform(df)

    scaled = pd.DataFrame(scaled, columns = df.columns)

    scaled.head()

    return scaled