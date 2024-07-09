import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler


def read_data(file_path):
    df = pd.read_csv(file_path, index_col="CUST_ID")
    df.columns = [col.lower() for col in df.columns]
    return df


def impute_missing_values(df):
    predictors = df.loc[~df['minimum_payments'].isna(), :].copy()
    predictors.dropna(inplace=True)
    target = predictors['minimum_payments'].copy()
    predictors.drop(columns=['minimum_payments'], inplace=True)

    lm1 = LinearRegression()
    lm1.fit(predictors, target)

    # create predictor matrix (table) by filtering data for cases where minimum_payments is null
    predictors_imputation = df.loc[df['minimum_payments'].isna(), :].drop(columns=[
        'minimum_payments'])
    # ^^^ and drop minimum_payments column
    # generate predictions
    imputed_minimum_payments_using_lm = lm1.predict(predictors_imputation)

    # overwrite the nulls in df with these 313 values
    df.loc[df['minimum_payments'].isna(
    ), 'minimum_payments'] = imputed_minimum_payments_using_lm
    return df


def scale_data(df):
    # instantiate the thing (in this case the robustscaler)
    scaler = RobustScaler()

    # fit the thing (our scaler)
    scaler.fit(df)

    # transform the data using the thing
    scaled = scaler.transform(df)
    scaled = pd.DataFrame(scaled, columns=df.columns)
    return scaled


def windsor_data(df):
    for col in df.columns:
        if np.max(df[col]) == 1.0:
            continue
        upper_limit = np.percentile(df[col], 99)
        lower_limit = np.min(df[col])
        df[col] = np.clip(df[col], lower_limit, upper_limit)
    return df


def scale_data1(df):
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns)
    return scaled_df


def impute_missing_values1(df):
    """
    Imputes missing values in the 'minimum_payments' column using a constant of proportionality method.
    Uses Linear Regression for imputation.
    """
 # Impute minimum_payments using constant of proportionality
    c = df['minimum_payments'].mean() / df['balance'].mean()
    imputed_min_payments = c * df.loc[df['minimum_payments'].isna(), 'balance']
    # df['constant_proportionality'] = df['minimum_payments'].copy()
    target = df.loc[(~df['minimum_payments'].isna()) & (
        ~df['credit_limit'].isna()), 'minimum_payments'].copy()
    # target.dropna(inplace = True)
#    df.loc[df['constant_proportionality'].isna(
#    ), 'constant_proportionality'] = imputed_min_payments
    predicted_minimum_payments_using_cop = c*df.loc[(~df['minimum_payments'].isna()) & (
        ~df['credit_limit'].isna()), 'balance']
    return target, predicted_minimum_payments_using_cop


def impute_missing_values_linear(df):
    predictors = df.loc[~df['minimum_payments'].isna(), :].copy()
    predictors.dropna(inplace=True)
    target = predictors['minimum_payments'].copy()
    predictors.drop(columns=['minimum_payments'], inplace=True)
    lm1 = LinearRegression()
    lm1.fit(predictors, target)
    # calculate r-squared
    return lm1.score(predictors, target)
