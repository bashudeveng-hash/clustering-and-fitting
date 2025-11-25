"""
This is the template file for the clustering and fitting assignment.
Completed fully and runnable using your dataset: data.csv.
"""

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
    """Create a scatter relational plot using first 2 numeric columns."""
    num_cols = df.select_dtypes(include=[float, int]).columns[:2]
    x, y = num_cols[0], num_cols[1]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Relational Plot ({x} vs {y})")
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Bar plot for the first column."""
    col = df.columns[0]
    plt.figure(figsize=(7, 5))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Categorical Plot of {col}")
    plt.savefig('categorical_plot.png')
    plt.close()
    return


def plot_statistical_plot(df):
    """Correlation heatmap of numeric columns."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Statistical Plot (Correlation Heatmap)")
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Return 4 statistical moments of a column."""
    series = df[col].astype(float)

    mean = series.mean()
    stddev = series.std()
    skew = ss.skew(series)
    excess_kurtosis = ss.kurtosis(series)   # excess kurtosis

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Basic preprocessing."""
    print("\nData Overview:")
    print(df.head())
    print("\nDescription:")
    print(df.describe())
    print("\nCorrelations:")
    print(df.corr(numeric_only=True))

    df = df.drop_duplicates()
    df = df.dropna()

    return df


def perform_clustering(df, col1, col2):
    """Full clustering workflow."""
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    inertias = []
    silhouettes = []

    # Elbow + silhouette
    for k in range(2, 6):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled, labels))

    # Elbow method plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 6), inertias, marker='o')
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig("elbow_plot.png")
    plt.close()

    # Best k from silhouette
    best_k = np.argmax(silhouettes) + 2
    final_model = KMeans(n_clusters=best_k, random_state=42)
    labels = final_model.fit_predict(scaled)

    centers_scaled = final_model.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    xkmeans = centers[:, 0]
    ykmeans = centers[:, 1]
    cenlabels = list(range(best_k))

    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Scatter plot of clusters."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels, palette="viridis")
    plt.scatter(xkmeans, ykmeans, s=200, c='red', label="Centers", marker="X")
    plt.title("Clustered Data")
    plt.legend()
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Linear regression for two chosen columns."""
    data = df[[col1, col2]].dropna()
    X = data[[col1]]
    y = data[col2]

    model = LinearRegression()
    model.fit(X, y)

    x_line = np.linspace(X.min(), X.max(), 200)
    y_line = model.predict(x_line)

    return data, x_line, y_line


def plot_fitted_data(data, x, y):
    """Plot regression fit."""
    plt.figure(figsize=(7, 5))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], label="Actual")
    plt.plot(x, y, color="red", linewidth=3, label="Fitted Line")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.savefig("fitting.png")
    plt.close()
    return


def writing(moments, col):
    mean, stddev, skew, kurt = moments

    print(f'\nFor the attribute "{col}":')
    print(f'Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, Excess Kurtosis = {kurt:.2f}.')

    # Interpretation
    skew_type = "not skewed"
    if skew > 2:
        skew_type = "right skewed"
    elif skew < -2:
        skew_type = "left skewed"

    kurt_type = "mesokurtic"
    if kurt > 2:
        kurt_type = "leptokurtic"
    elif kurt < -2:
        kurt_type = "platykurtic"

    print(f"The data was {skew_type} and {kurt_type}.")
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    # select first numeric column automatically
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    col = numeric_cols[0]

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    # clustering using first 2 numeric columns
    clustering_results = perform_clustering(
        df,
        numeric_cols[0],
        numeric_cols[1]
    )
    plot_clustered_data(*clustering_results)

    # fitting using next 2 numeric columns
    fitting_results = perform_fitting(
        df,
        numeric_cols[1],
        numeric_cols[2]
    )
    plot_fitted_data(*fitting_results)

    return


if __name__ == '__main__':
    main()
