import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# ==========================
#   Plotting Constants
# ==========================
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
LINESTYLES = ["-", ":", "--", "-."]
FIG_WIDTH = 14.39 / 2.54
FIG_HEIGHT = 12.09 / 2.54

fontsize_xy_label = 14
fontsize_ticks = 12
fontsize_legend = 12
fontsize_text = 8


def setup_directories():
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def load_and_split_data(filepath):
    df = pd.read_csv(filepath)

    target = df.columns[0]
    features = df.columns[1:]

    x = df.loc[:, features].to_numpy()
    y = df.loc[:, target].to_numpy()

    return train_test_split(x, y, test_size=0.2, random_state=0), features


def optimize_models(x_train, y_train):
    poly_pipe = make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression())
    knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())

    kwargs_gscv = dict(cv=5, scoring="neg_root_mean_squared_error", return_train_score=True)

    param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 5)])
    param_grid_knn = dict(
        kneighborsregressor__n_neighbors=[i for i in range(1, 81)],
        kneighborsregressor__weights=["uniform", "distance"]
    )
    # --- Polynomial Regression Optimization ---
    print("Optimizing Polynomial Regression...")
    poly_grid = GridSearchCV(poly_pipe, param_grid=param_grid_poly, **kwargs_gscv)
    poly_grid.fit(x_train, y_train)

    df_poly = pd.DataFrame(poly_grid.cv_results_).sort_values(by=['rank_test_score'], ascending=True)
    columns_view_poly = [
        "rank_test_score", "param_polynomialfeatures__degree",
        "mean_test_score", "std_test_score", "mean_train_score", "std_train_score"
    ]
    print("\nRecall that we used negative RMSE, therefore a higher score (i.e., closer to zero) is better:\n")
    print(df_poly.loc[:, columns_view_poly].to_string())
    print()

    # --- kNN Optimization ---
    print("Optimizing k-Nearest Neighbors...")
    knn_grid = GridSearchCV(knn_pipe, param_grid=param_grid_knn, **kwargs_gscv)
    knn_grid.fit(x_train, y_train)

    df_knn = pd.DataFrame(knn_grid.cv_results_).sort_values(by=['rank_test_score'], ascending=True)
    columns_view_knn = [
        "rank_test_score", "param_kneighborsregressor__n_neighbors", "param_kneighborsregressor__weights",
        "mean_test_score", "std_test_score", "mean_train_score", "std_train_score"
    ]
    print("\nRecall that we used negative RMSE, therefore a higher score (i.e., closer to zero) is better:\n")
    print(df_knn.loc[:, columns_view_knn].to_string())
    print()

    return poly_grid, knn_grid, param_grid_knn


def evaluate_and_export_results(poly_grid, knn_grid, x_test, y_test):
    best_poly = poly_grid.best_estimator_
    best_knn = knn_grid.best_estimator_

    print(f"Final Poly Test negative RMSE: {poly_grid.score(x_test, y_test):.6f} d^-1")
    print(f"Final kNN Test negative RMSE: {knn_grid.score(x_test, y_test):.6f} d^-1\n")

    models = {
        f"Polynomial Regression (Degree {poly_grid.best_params_['polynomialfeatures__degree']})": best_poly,
        f"kNN (k={knn_grid.best_params_['kneighborsregressor__n_neighbors']}, weights={knn_grid.best_params_['kneighborsregressor__weights']})": best_knn
    }

    results = []
    for name, model in models.items():
        y_pred = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "RMSE": round(rmse, 4), "R2_Score": round(r2, 4)})

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join("tables", "model_comparison_summary.csv"), index=False)
    print("Exported model comparison to tables/model_comparison_summary.csv")
    return best_poly


def export_feature_importance(best_poly, features):
    poly_features = best_poly.named_steps['polynomialfeatures'].get_feature_names_out(features)
    poly_weights = best_poly.named_steps['linearregression'].coef_

    df_weights = pd.DataFrame({
        "Feature": poly_features,
        "Weight": poly_weights,
        "Abs_Weight": np.abs(poly_weights)
    }).sort_values(by="Abs_Weight", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained")
    y_pos = np.arange(len(df_weights))
    colors = ['tab:green' if w > 0 else 'tab:red' for w in df_weights["Weight"]]

    ax.barh(y_pos, df_weights["Weight"], align='center', color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos, labels=df_weights["Feature"])

    ax.set_xlabel('Standardized Model Weight', fontsize=fontsize_xy_label)
    ax.set_title('Top 10 Feature Weights (Polynomial)', fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    fig.savefig(os.path.join("figures", "model_weights.png"), dpi=200)
    plt.close(fig)


def export_scaling_comparison(X_train, features):
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5), layout="constrained")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    markers = ['o', 's', '^', 'D']

    for i, feature in enumerate(features):
        axes[0].scatter(X_train[:, i], np.full(X_train.shape[0], i),
                        label=feature, alpha=0.3, marker=markers[i], color=COLORS[i], edgecolor='black')
        axes[1].scatter(X_train_scaled[:, i], np.full(X_train_scaled.shape[0], i),
                        label=feature, alpha=0.3, marker=markers[i], color=COLORS[i], edgecolor='black')

    axes[0].set_title("Raw Data Features", fontsize=fontsize_xy_label)
    axes[0].set_xlabel("Feature Value", fontsize=fontsize_xy_label)
    axes[0].set_yticks([])
    axes[0].legend(fontsize=fontsize_text, loc='upper right')

    axes[1].set_title("Standardized Data Features", fontsize=fontsize_xy_label)
    axes[1].set_xlabel("Feature Value (Standard Deviations)", fontsize=fontsize_xy_label)
    axes[1].set_yticks([])

    fig.savefig(os.path.join("figures", "data_scaling_comparison.png"), dpi=200)
    plt.close(fig)


def export_validation_curve(poly_grid):
    df = pd.DataFrame(poly_grid.cv_results_)
    plot_n = 4
    degrees = df.loc[:, "param_polynomialfeatures__degree"].to_numpy()
    rmse_train = - df.loc[:, "mean_train_score"].to_numpy()
    rmse_test = - df.loc[:, "mean_test_score"].to_numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

    ax.plot(degrees[:plot_n], rmse_train[:plot_n], label=f"Train (Poly)", color=COLORS[0], linestyle="-")
    ax.plot(degrees[:plot_n], rmse_test[:plot_n], label=f"Valid (Poly)", color=COLORS[0], linestyle="--")
    ax.set_xlabel("Polynomial Degree", fontsize=fontsize_xy_label)
    ax.set_ylabel("Specific Growth Rate RMSE [d^-1]", fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.legend(ncols=1, fontsize=fontsize_legend)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    fig.savefig(os.path.join("figures", "poly_validation_curve.png"), dpi=200)
    plt.close(fig)


def export_knn_validation_curve(knn_grid, param_grid_knn):
    df = pd.DataFrame(knn_grid.cv_results_)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")

    for i, weight in enumerate(param_grid_knn["kneighborsregressor__weights"], start=1):
        mask_weight = (df.loc[:, "param_kneighborsregressor__weights"] == weight)
        df_sub = df.loc[mask_weight, :]
        df_sub = df_sub.sort_values(by=['param_kneighborsregressor__n_neighbors'], ascending=True)

        n_neighbors = df_sub.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
        rmse_train = - df_sub.loc[:, "mean_train_score"].to_numpy()
        rmse_test = - df_sub.loc[:, "mean_test_score"].to_numpy()

        ax.plot(n_neighbors, rmse_train, label=f"Train (kNN: {weight})", color=COLORS[i], linestyle="-")
        ax.plot(n_neighbors, rmse_test, label=f"Valid (kNN: {weight})", color=COLORS[i], linestyle="--")

    ax.set_xlabel("k", fontsize=fontsize_xy_label)
    ax.set_ylabel("Specific Growth Rate RMSE [d^-1]", fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.legend(ncols=1, fontsize=fontsize_legend)
    ax.set_ylim(bottom=0)

    fig.savefig(os.path.join("figures", "knn_validation_curve.png"), dpi=200)
    plt.close(fig)


def export_temperature_curve(best_poly, best_knn, X_train, features):
    means = np.mean(X_train, axis=0)
    temp_idx = list(features).index("Temperature [C]")

    t_min, t_max = np.min(X_train[:, temp_idx]), np.max(X_train[:, temp_idx])
    t_plot = np.linspace(t_min, t_max, 100)

    X_plot = np.tile(means, (100, 1))
    X_plot[:, temp_idx] = t_plot

    y_poly = best_poly.predict(X_plot)
    y_knn = best_knn.predict(X_plot)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained")
    ax.plot(t_plot, y_poly, label="Model (Poly)", color=COLORS[0], linestyle=":", linewidth=2)
    ax.plot(t_plot, y_knn, label="Model (kNN)", color=COLORS[1], linestyle="--", linewidth=2)

    ax.set_xlabel("Temperature [C]", fontsize=fontsize_xy_label)
    ax.set_ylabel("Specific Growth Rate [d^-1]", fontsize=fontsize_xy_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    ax.legend(fontsize=fontsize_xy_label)

    fig.savefig(os.path.join("figures", "temperature_dependence_curve.png"), dpi=200)
    plt.close(fig)