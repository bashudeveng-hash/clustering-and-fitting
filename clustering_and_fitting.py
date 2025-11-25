import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    num_cols = df.select_dtypes(include=[float, int]).columns
    if len(num_cols) < 2:
        return
    x = num_cols[0]
    y = num_cols[1]
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Relational Plot ({x} vs {y})")
    plt.savefig("relational_plot.png")
    plt.close()


def plot_categorical_plot(df):
    col = df.columns[0]
    plt.figure(figsize=(7, 5))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Categorical Plot of {col}")
    plt.savefig("categorical_plot.png")
    plt.close()


def plot_statistical_plot(df):
    corr = df.corr(numeric_only=True)
    if corr.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Statistical Plot")
    plt.savefig("statistical_plot.png")
    plt.close()


def statistical_analysis(df, col):
    series = df[col].astype(float)
    mean = series.mean()
    stddev = series.std()
    skew = ss.skew(series)
    kurt = ss.kurtosis(series)
    return mean, stddev, skew, kurt


def writing(moments, col):
    mean, stddev, skew, kurt = moments
    print(f"For '{col}': mean={mean:.2f}, std={stddev:.2f}, skew={skew:.2f}, kurt={kurt:.2f}")


def preprocessing(df):
    print(df.head())
    print(df.describe(include="all"))
    print(
