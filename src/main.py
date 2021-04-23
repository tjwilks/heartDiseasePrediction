import configparser
from src.utils import read_input_data, preprocessing
from src.modelling import total_grid_search


def main():
    # read in package config
    config = configparser.ConfigParser()
    config.read("config/config.yml")

    # read in input data
    all_data = read_input_data.get_data(
        config["read_input_data_config"]
    )

    # preprocess input data
    data_preprocessor = preprocessing.DataPreprocessor(all_data)
    data_preprocessor.preprocess_data()

    # run hyperparameter search
    hyp_param_search = total_grid_search.HypParamSearch(
        data_preprocessor.X_data,
        data_preprocessor.y_data,
        col_na_proportion=data_preprocessor.col_na_proportion
    )
    grid_search_cv_preprocess = hyp_param_search.grid_hyp_search(config["grid_search_config"])
    bayes_search_cv_preprocess = hyp_param_search.bayes_opt_hyp_search(config["bayes_opt_search_config"])

    # print hyperparameter search results
    total_grid_search.get_grid_search_results(grid_search_cv_preprocess)
    total_grid_search.get_grid_search_results(bayes_search_cv_preprocess)

if __name__ == "__main__":
    main()

