import configparser
from src.utils import read_input_data, preprocessing
from src.modelling import random_forrest_modelling
import numpy as np

def main():
    config = configparser.ConfigParser()
    config.read("config.yml")
    data = read_input_data.get_data(
        config["default"].get("heart_disease_data_dir"),
        ["switzerland.data",
         "long-beach-va.data",
         "hungarian.data"]
    )
    data_preprocessor = preprocessing.DataPreprocessor(data, 0.01)
    data_preprocessor.preprocess_data()
    baseline_rf_model = random_forrest_modelling.train_random_forrests(
        data_preprocessor.X_data,
        data_preprocessor.y_data
    )
    feature_importance_dict = random_forrest_modelling.random_forrest_feature_importances(
        baseline_rf_model,
        data_preprocessor.features
    )
    selected_features = random_forrest_modelling.feature_importance_feature_selection(
        feature_importance_dict,
        50
    )
    X_data = random_forrest_modelling.filter_X_data_to_selected_features(
        selected_features,
        data_preprocessor.features,
        data_preprocessor.X_data
    )
    baseline_rf_model = random_forrest_modelling.train_random_forrests(
        X_data,
        data_preprocessor.y_data
    )
    grid_search_cv_rf = random_forrest_modelling.random_forrest_grid_search(
        X_data,
        data_preprocessor.y_data
    )
    random_forrest_modelling.grid_search_cv_rf_results(grid_search_cv_rf)

if __name__ == "__main__":
    main()

