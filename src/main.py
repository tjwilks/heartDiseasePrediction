import configparser
from src.utils import read_input_data, preprocessing
from src.modelling import total_grid_search

def main():
    config = configparser.ConfigParser()
    config.read("config.yml")
    X_data, y_data = read_input_data.get_data(
        config["default"].get("heart_disease_data_dir"),
        ["switzerland.data",
         "long-beach-va.data",
         "hungarian.data"]
    )
    data_preprocessor = preprocessing.DataPreprocessor(X_data, y_data, 0.8, 8)
    data_preprocessor.preprocess_data()

    fit_params = {"col_na_proportion": data_preprocessor.col_na_proportion}
    grid_search_cv_rf = total_grid_search.total_grid_search(
        data_preprocessor.X_data,
        data_preprocessor.y_data,
        missing_value_filter=[0.8, 0.9],
        feature_importance_rank_filter=[10, 15],
        max_leaf_nodes=[20, 30],
        max_features=[5, 7],
        n_estimators=[100],
        fit_params=fit_params,
    )
    total_grid_search.get_grid_search_results(grid_search_cv_rf)

if __name__ == "__main__":
    main()

