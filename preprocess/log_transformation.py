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
    num_cols = len(df_transformed[highly_skewed_columns].columns)
    fig, axs = plt.subplots(nrows=num_cols, ncols=2, figsize=(20, 40))
    fig.subplots_adjust(hspace=0.4)

    for i, col in enumerate(df_transformed[highly_skewed_columns].columns):
    
        sns.histplot(data=df, x=col, ax=axs[i, 0], bins=20, color='blue', alpha=0.5)
        axs[i, 0].set_title(f'Original {col} Distribution')
    
        sns.histplot(data=df_transformed, x=col, ax=axs[i, 1], bins=20, color='green', alpha=0.5)
        axs[i, 1].set_title(f'Log Transformed {col} Distribution')
    plt.tight_layout()
    plt.show()

    return df_transformed