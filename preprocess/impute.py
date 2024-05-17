# code that imputes missing values into the data
from sklearn.linear_model import LinearRegression

def impute_drop_missing_values(df):

    predictors = df.loc[~df['minimum_payments'].isna(),:]
    predictors.dropna(inplace=True)
    target = predictors['minimum_payments']
    predictors.drop(columns = ['minimum_payments'],inplace = True)

    lm1 = LinearRegression()
    lm1.fit(predictors,target)
    lm1.score(predictors,target)

    predictors_imputation = df.loc[df['minimum_payments'].isna(),:].drop(columns = ['minimum_payments'])
    imputed_minimum_payments_using_lm = lm1.predict(predictors_imputation)
    df.loc[df['minimum_payments'].isna(),'minimum_payments'] = imputed_minimum_payments_using_lm

    df.dropna(inplace=True)

    return df

