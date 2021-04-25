from src.utils import read_input_data
from src.utils.preprocessing import DataPreprocessor
from src.modelling.total_grid_search import TotalEstimator, HypParamSearch
import configparser
from sklearn import utils
import pytest
import numpy as np


class TestTotalEstimator:
    """
        a class for testing the TotalEstimator class

        Methods
        -------
        setup_test:
            initialises and fits TotalEstimator object. Prepares
            X and y data.

        test_is_fitted_estimator:
            tests if the estimator is fitted

        test_is_valid_input:
            Tests:
             - X and y are same length
             - X is 2D and y 1D
             - X and y are not empty
             - X and y do not contain np.nan or np.inf

        tests_predict:
            tests that predicted unique values are identical to y
            unique values and that number of predictions is equal to
            number of y values

        tests_score:
            tests that score is between 0 and 1
    """

    @pytest.fixture(autouse=True)
    def setup_test(self):
        self.config = configparser.ConfigParser()
        self.config.read("config/config.yml")
        self.config = self.config["read_input_data_config"]
        self.data = read_input_data.get_data(self.config)
        self.data_preprocessor = DataPreprocessor(self.data)
        self.data_preprocessor.preprocess_data()
        self.total_estimator = TotalEstimator(missing_value_filter=0.5,
                                              feature_importance_rank_filter=5,
                                              n_estimators=200,
                                              max_features=8,
                                              max_leaf_nodes=25,
                                              fit_params=None,
                                              times_fit=1)
        self.X = self.data_preprocessor.X_data
        self.y = self.data_preprocessor.y_data
        fit_params = {"col_na_proportion": self.data_preprocessor.col_na_proportion}
        self.fitted_estimator = self.total_estimator.fit(self.X, self.y, **fit_params)

    def test_is_fitted_estimator(self):
        def is_fiited(estimator):
            try:
                utils.validation.check_is_fitted(estimator)
                return True
            except:
                return False

        assert is_fiited(self.fitted_estimator)

    def test_is_valid_input(self):
        def is_valid_input(X, y):
            try:
                utils.validation.check_X_y(X, y)
                return True
            except:
                return False

        assert is_valid_input(self.X, self.y)  # before fit applied
        assert is_valid_input(self.fitted_estimator.X_, self.fitted_estimator.y_)  # after fit applied

    def test_predict(self):
        predictions = self.fitted_estimator.predict(self.X)
        assert set(self.y) == set(predictions)
        assert len(self.y) == len(predictions)

    def test_score(self):
        score = self.fitted_estimator.score(self.X, self.y)
        assert score <= 1
        assert score >= 0


class TestHypParamSearch:
    """
        a class for testing the HypParamSearch class

        Methods
        -------
        setup_test:
            initialises HypParamSearch object

        test_grid_hyp_search_returns_all_configs:
            tests number of results available after grid search
            is equal to the number of results requested in
            grid_config_param_grid.json

        test_bayes_opt_search_returns_n_iterations:
            tests number of results available after bayes opt
            search is equal to number of iterations requested
            in config.yml

    """
    @pytest.fixture(autouse=True)
    def setup_test(self):
        self.config = configparser.ConfigParser()
        self.config.read("config/config.yml")
        self.read_input_config = self.config["read_input_data_config"]
        self.data = read_input_data.get_data(self.read_input_config)
        self.data_preprocessor = DataPreprocessor(self.data)
        self.data_preprocessor.preprocess_data()
        self.hyp_param_search = HypParamSearch(
            self.data_preprocessor.X_data,
            self.data_preprocessor.y_data,
            col_na_proportion=self.data_preprocessor.col_na_proportion
        )

    def test_grid_hyp_search_returns_all_configs(self):
        grid_search_config = self.config["grid_search_config"]
        grid_search_cv_preprocess = self.hyp_param_search.grid_hyp_search(grid_search_config)
        grid_search_param_grid = read_input_data.read_json_to_dict(grid_search_config["param_grid_config_path"])
        grid_configurations = [len(np.arange(value[0], value[1], value[2]))
                               for value in grid_search_param_grid.values()]
        total_configurations = np.prod(grid_configurations)
        len_results = len(grid_search_cv_preprocess.cv_results_["mean_test_score"])
        assert len_results == total_configurations

    def test_bayes_opt_search_returns_n_iterations(self):
        bayes_opt_search_config = self.config["bayes_opt_search_config"]
        grid_search_cv_preprocess = self.hyp_param_search.bayes_opt_hyp_search(bayes_opt_search_config)
        len_results = len(grid_search_cv_preprocess.cv_results_["mean_test_score"])
        n_iterations = bayes_opt_search_config.getint("iterations")
        assert len_results == n_iterations
