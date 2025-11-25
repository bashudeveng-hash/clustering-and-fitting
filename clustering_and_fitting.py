"""
Final corrected version — fully runnable with data.csv
Improvements:
✔ Safe numeric column handling
✔ Fixed linear regression linspace shape error
✔ Improved clustering (n_init, labels)
✔ Added error handling
✔ Improved interpretive writing
✔ Cleaner plots
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


# ----------------------------------------------------
# BASIC PLOTTING FUNCTIONS
# ----------------------------------------------------

def plot_relational_plot(df):
    """Scatter plot using first 2 numeric columns."""
    num_cols = df.select_dtypes(include=[float, int]).columns
    if len(num_cols) < 2:
        print("Not enough numeric columns for relational plot.")
        return

    x, y = num_cols[0], num_cols[1]

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f"Relational Plot ({x} vs {y})")
    plt.savefig('relational_plot.png')
    plt.close()


def plot_categorical_plot(df):
    """Bar plot for the first column."""
    col = df.columns[0]
    plt.figure(figsize=(7, 5))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Categorical Plot of {col}")
    plt.savefig('categorical_plot.png')
    plt.close()


def plot_statistical_plot(df):
    """Correlation heatmap of numeric columns."""
    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)

    if corr.empty:
        print("No numeric columns for correlation heatmap.")
        return

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Statistical Plot (Correlation Heatmap)")
    plt.savefig('statistical_plot.png')
    plt.close()


# ----------------------------------------------------
# STATISTICAL ANALYSIS
# ----------------------------------------------------

def statistical_analysis(df, col):
    """Return four statistical moments of a column."""
    series = df[col].astype(float)

    mean = series.mean()
    stddev = series.std()
    skew = ss.skew(series)
    kurt = ss.kurtosis(series)  # excess kurtosis

    return mean, stddev, skew, kurt


def writing(moments, col):
    mean, stddev, skew, kurt = moments

    print(f'\nFor the attribute "{col}":')
    print(f'Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, Excess Kurtosis = {kurt:.2f}.')

    # Interpretation
    if abs(skew) < 0.5:
        skew_type = "approximately symmetric"
    elif skew > 0.5:
        skew_type = "right skewed"
    else:
        skew_type = "left skewed"

    if kurt > 2:
        kurt_type = "leptokurtic (sharper peak)"
    elif kurt < -2:
        kurt_type = "platykurtic (flatter distribution)"
    else:
        kurt_type = "mesokurtic (normal-like)"

    print(f"The data is {skew_type} and {kurt_type}.")


# ----------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------

def preprocessing(df):
    """Basic data cleaning."""
    print("\nData Overview:")
    print(df.head())

    print("\nDescription:")
    print(df.describe(include='all'))

    print("\nCorrelations:")
    print(df.corr(numeric_only=True))

    df = df.drop_duplicates()
    df = df.dropna()

    return df


# ----------------------------------------------------
# CLUSTERING
# ----------------------------------------------------

def perform_clustering(df, col1, col2):
    """Perform KMeans clustering using two numeric columns."""
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    inertias = []
    silhouettes = []

    # Try cluster sizes from 2 to 5
    for k in range(2, 6):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)

        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled, labels))

    # Save elbow plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 6), inertias, marker='o')
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig("elbow_plot.png")
    plt.close()

    # Best k
    best_k = silhouettes.index(max(silhouettes)) + 2
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(scaled)

    centers_scaled = final_model.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)

    xkmeans = centers[:, 0]
    ykmeans = centers[:, 1]
    cenlabels = list(range(best_k))

    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot clusters and centroids."""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1],
                    hue=labels, palette="viridis")

    plt.scatter(xkmeans, ykmeans, s=200, color='red', label="Centers", marker="X")

    for i, (cx, cy) in enumerate(zip(xkmeans, ykmeans)):
        plt.text(cx, cy, f"C{i}", fontsize=12, color="red")

    plt.title("Clustered Data")
    plt.legend()
    plt.savefig('clustering.png')
    plt.close()


# ----------------------------------------------------
# REGRESSION FITTING
# ----------------------------------------------------

def perform_fitting(df, col1, col2):
    """Perform linear regression fitting."""
    data = df[[col1, col2]].dropna()
    X = data[[col1]]
    y = data[col2]

    model = LinearRegression()
    model.fit(X, y)

    x_line = np.linspace(X[col1].min(), X[col1].max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    return data, x_line, y_line


def plot_fitted_data(data, x, y):
    """Plot regression line."""
    plt.figure(figsize=(7, 5))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], label="Actual Data")
    plt.plot(x, y, color="red", linewidth=3, label="Fitted Line")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.savefig("fitting.png")
    plt.close()


# ----------------------------------------------------
# MAIN FUNCTION
# ----------------------------------------------------

def main():
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("ERROR: data.csv not found!")
        return

    df = preprocessing(df)

    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    if len(numeric_cols) < 3:
        print("ERROR: Need at least 3 numeric columns for clustering & fitting.")
        return

    # Plots
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    # Statistical Analysis
    moments = statistical_analysis(df, numeric_cols[0])
    writing(moments, numeric_cols[0])

    # Clustering
    labels, data, xkmeans, ykmeans, cenlabels = perform_clustering(
        df, numeric_cols[0], numeric_cols[1]
    )
    plot_clustered_data(labels, data, xkmeans, ykmeans, cenlabels)

    # Fitting
    data_fit, x_line, y_line = perform_fitting(
        df, numeric_cols[1], numeric_cols[2]
    )
    plot_fitted_data(data_fit, x_line, y_line)


if __name__ == '__main__':
    main()
