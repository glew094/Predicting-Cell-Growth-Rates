import os
from src.functions import (
    setup_directories,
    load_and_split_data,
    optimize_models,
    evaluate_and_export_results,
    export_feature_importance,
    export_scaling_comparison,
    export_validation_curve,
    export_knn_validation_curve,
    export_temperature_curve
)


def main():
    setup_directories()

    filepath = os.path.join("datasets", "dataset_cell_growth_4factor.csv")
    (x_train, x_test, y_train, y_test), features = load_and_split_data(filepath)

    poly_grid, knn_grid, param_grid_knn = optimize_models(x_train, y_train)

    best_poly = evaluate_and_export_results(poly_grid, knn_grid, x_test, y_test)

    export_feature_importance(best_poly, features)
    export_scaling_comparison(x_train, features)
    export_validation_curve(poly_grid)
    export_knn_validation_curve(knn_grid, param_grid_knn)
    export_temperature_curve(best_poly, knn_grid.best_estimator_, x_train, features)


if __name__ == "__main__":
    main()