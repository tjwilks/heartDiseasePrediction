import configparser
from src.utils import read_input_data, preprocessing
from src.modelling import feature_selection, random_forrest_modelling

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
    baseline_rf_model = random_forrest_modelling.train_random_forrests(
        data_preprocessor.X_data,
        data_preprocessor.y_data
    )
    fit_params = {"col_na_proportion": data_preprocessor.col_na_proportion}
    grid_search_cv_rf = feature_selection.feature_selection_grid_search(
        data_preprocessor.X_data,
        data_preprocessor.y_data,
        fit_params
    )
    feature_selection.get_feature_selection_grid_search_results(grid_search_cv_rf)
    grid_search_cv_rf = random_forrest_modelling.random_forrest_grid_search(
        data_preprocessor.X_data,
        data_preprocessor.y_data
    )
    random_forrest_modelling.grid_search_cv_rf_results(grid_search_cv_rf)

if __name__ == "__main__":
    main()

