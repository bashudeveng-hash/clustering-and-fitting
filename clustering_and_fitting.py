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
# PLOTTING
# ----------------------------------------------------

def plot_relational_plot(df):
    num_cols = df.select_dtypes(include=[float, int]).columns
    if len(num_cols) < 2:
        return

    x, y = num_cols[0], num_cols[1]

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
    plt.title("Correlation Heatmap")
    plt.savefig("statistical_plot.png")
    plt.close()


# ----------------------------------------------------
# STATISTICS
# ----------------------------------------------------

def statistical_analysis(df, col):
    series = df[col].astype(float)

    mean = series.mean()
    stddev = series.std()
    skew = ss.skew(series)
    kurt = ss.kurtosis(series)

    return mean, stddev, skew, kurt


def writing(moments, col):
    mean, stddev, skew, kurt = moments

    print(f"\nFor the attribute '{col}':")
    print(f"Mean = {mean:.2f}, StdDev = {stddev:.2f}, "
          f"Skew = {skew:.2f}, Kurtosis = {kurt:.2f}")


# ----------------------------------------------------
# PREPROCESS
# ----------------------------------------------------

def preprocessing(df):
    print(df.head())
    print(df.describe(include='all'))
    print(df.corr(numeric_only=True))

    df = df.drop_duplicates()
    df = df.dropna()
    return df


# ----------------------------------------------------
# CLUSTERING
# ----------------------------------------------------

def perform_clustering(df, col1, col2):
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    best_k = 2
    best_score = -1

    for k in range(2, 6):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        score = silhouette_score(scaled, labels)

        if score > best_score:
            best_score = score
            best_k = k

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(scaled)

    centers = scaler.inverse_transform(final_model.cluster_centers_)
    xk = centers[:, 0]
    yk = centers[:, 1]

    return labels, data, xk, yk, range(best_k)


def plot_clustered_data(labels, data, xkmeans, ykmeans, cenlabels):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue=labels)
    plt.scatter(xkmeans, ykmeans, color='red', s=200, marker="X")

    plt.title("Clustered Data")
    plt.savefig("clustering.png")
    plt.close()


# ----------------------------------------------------
# REGRESSION
# ----------------------------------------------------

def perform_fitting(df, col1, col2):
    data = df[[col1, col2]].dropna()
    X = data[[col1]]
    y = data[col2]

    model = LinearRegression()
    model.fit(X, y)

    x_line = np.linspace(X[col1].min(), X[col1].max(), 200).reshape(-1, 1)
    y_line = model.predict(x_line)

    return data, x_line, y_line


def plot_fitted_data(data, x, y):
    plt.figure(figsize=(7, 5))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.plot(x, y, color="red")
    plt.title("Linear Regression Fit")
    plt.savefig("fitting.png")
    plt.close()


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("ERROR: data.csv not found!")
        return

    df = preprocessing(df)

    num = df.select_dtypes(include=[float, int]).columns.tolist()
    if len(num) < 3:
        print("Need at least 3 numeric columns.")
        return

    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)

    moments = statistical_analysis(df, num[0])
    writing(moments, num[0])

    labels, data, xk, yk, cen = perform_clustering(df, num[0], num[1])
    plot_clustered_data(labels, data, xk, yk, cen)

    data_fit, x_line, y_line = perform_fitting(df, num[1], num[2])
    plot_fitted_data(data_fit, x_line, y_line)


if __name__ == "__main__":
    main()
