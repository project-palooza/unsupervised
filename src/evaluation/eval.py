import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_imputation(df):
    # R squared for constant of proportionality
    c = df['minimum_payments'].mean() / df['balance'].mean()
    predicted_minimum_payments_using_cop = c*df.loc[(~df['minimum_payments'].isna()) & (
        ~df['credit_limit'].isna()), 'balance']

    # r-squared for linear regression
    predictors = df.loc[~df['minimum_payments'].isna(), :].copy()
    predictors.dropna(inplace=True)
    target = predictors['minimum_payments'].copy()
    predictors.drop(columns=['minimum_payments'], inplace=True)
    lm1 = LinearRegression()
    lm1.fit(predictors, target)
    return r2_score(target, predicted_minimum_payments_using_cop), lm1.score(predictors, target)


def r_squared_constant(df):
    c = df['minimum_payments'].mean() / df['balance'].mean()
    target = df.loc[(~df['minimum_payments'].isna()) & (
        ~df['credit_limit'].isna()), 'minimum_payments'].copy()
    predicted_minimum_payments_using_cop = c*df.loc[(~df['minimum_payments'].isna()) & (
        ~df['credit_limit'].isna()), 'balance']
    return r2_score(target, predicted_minimum_payments_using_cop)


def r_squared_linear(df):
    predictors = df.loc[~df['minimum_payments'].isna(), :].copy()
    predictors.dropna(inplace=True)
    target = predictors['minimum_payments'].copy()
    predictors.drop(columns=['minimum_payments'], inplace=True)
    lm1 = LinearRegression()
    lm1.fit(predictors, target)
    # calculate r-squared
    return lm1.score(predictors, target)


def plot_scores(k_range, scores, score_type):
    plt.figure()
    plt.plot(k_range, scores, marker="X")
    plt.title(f"{score_type} as a function of K (number of clusters)")
    plt.ylabel(score_type)
    plt.xlabel('k')
    plt.show()


def evaluate_clusters(k_range, scores, score_type):
    best_k = k_range[np.argmax(scores)]
    print(f"Best K based on {score_type}: {best_k}")


def compare_models(labels1, labels2):
    print("original labels")
    print(pd.crosstab(labels2, labels1,
                      rownames=['agglom'], colnames=['kmeans']))
    print("table with the kmeans labels flipped (0 -> 1 and vice versa)")
    print(pd.crosstab(labels2, (~labels1.astype(
        bool)).astype(int), rownames=['agglom'], colnames=['kmeans']))
    # table = pd.crosstab(labels1, labels2, rownames=[
    #                    'Method 1'], colnames=['Method 2'])
    # print("Agreement Table:")
    # print(table)


def column_compare(df):
    # Compare columns
    for col in [col for col in df.columns if col != "kmeans"]:
        balance_diff = df.groupby(['kmeans']).describe()[col].T
        balance_diff['ratio'] = balance_diff[1] / balance_diff[0]
        ratio = np.round(
            balance_diff.iloc[balance_diff.index == '50%', :]['ratio'].iloc[0])
        # balance_diff.loc[balance_diff.index == '50%', 'ratio'][0], 2)
        print(
            f"For {col}, the ratio of medians between group 1 and group 0 is: {ratio}")

    # Boxplots
    plt.subplots(9, 2, figsize=(15, 30))
    for i, col in enumerate([col for col in df.columns if col not in ['kmeans', 'agglom']]):
        plt.subplot(9, 2, i + 1)
        sns.boxplot(data=df, x="kmeans", y=col)
        plt.xlabel("")
    plt.show()
