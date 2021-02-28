import configparser
from src.utils import read_input_data, preprocess_data
from src.modelling import random_forrest_modelling


def main():
    config = configparser.ConfigParser()
    config.read("config.yml")
    data = read_input_data.get_data(
        config["default"].get("heart_disease_data_dir"),
        ["switzerland.data",
         "long-beach-va.data",
         "hungarian.data"]
    )
    features, X_data, y_data = preprocess_data.preprocess_data(data)
    baseline_rf_model = random_forrest_modelling.train_random_forrests(X_data, y_data)
    feature_importance_dict = random_forrest_modelling.random_forrest_feature_importances(baseline_rf_model, features)
    selected_features = random_forrest_modelling.feature_importance_feature_selection(feature_importance_dict, 50)
    X_data = random_forrest_modelling.filter_X_data_to_selected_features(selected_features, features, X_data)
    baseline_rf_model = random_forrest_modelling.train_random_forrests(X_data, y_data)
    grid_search_cv_rf = random_forrest_modelling.random_forrest_grid_search(X_data, y_data)
    random_forrest_modelling.grid_search_cv_rf_results(grid_search_cv_rf)

if __name__ == "__main__":
    main()

