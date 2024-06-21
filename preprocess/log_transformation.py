import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def right_skewed_data(df):
    # Identify highly skewed columns >1
    skewness = df.skew()
    highly_skewed_columns = skewness[skewness > 1].index
    df_transformed = df.copy()
    # Add 1 to avoid log(0) which is undefined
    df_transformed[highly_skewed_columns] = df_transformed[highly_skewed_columns].apply(lambda x: np.log1p(x))

    print("Skewness of each column:")
    print(df.skew())
    print("\nSkewness of each column after log transformation")
    print(df_transformed.skew())

    print("\nDistributions of each right-skewed column  before and after log transformation")
    fig, axes = plt.subplots(len(highly_skewed_columns), 2, figsize=(20, 40))

    for i, col in enumerate(highly_skewed_columns):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'Original {col} Distribution')
        sns.histplot(df_transformed[col], bins=30, kde=True, ax=axes[i, 1])
        axes[i, 1].set_title(f'Log Transformed {col} Distribution')

    plt.tight_layout()
    plt.show()

    return df_transformed