from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from src.modelling import feature_selection
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
import numpy as np
from src.utils import read_input_data


class HypParamSearch:

    def __init__(self, X_data, y_data, col_na_proportion):
        self.X_data = X_data
        self.y_data = y_data
        self.fit_params = {"col_na_proportion": col_na_proportion}

    def grid_hyp_search(self, grid_search_config):
        custom_estimator = TotalEstimator()
        grid_search_param_grid = read_input_data.read_json_to_dict(grid_search_config["param_grid_config_path"])
        grid_search_param_grid = {key: list(np.arange(value[0], value[1], value[2]))
                                  for key, value in grid_search_param_grid.items()}
        grid_search_cv_preprocess = GridSearchCV(custom_estimator,
                                                 grid_search_param_grid,
                                                 cv=grid_search_config.getint("k_folds"))
        grid_search_cv_preprocess.fit(self.X_data, self.y_data, fit_params=self.fit_params)
        return grid_search_cv_preprocess

    def bayes_opt_hyp_search(self, bayes_opt_search_config):
        custom_estimator = TotalEstimator()
        grid_search_param_grid = read_input_data.read_json_to_dict(bayes_opt_search_config["param_grid_config_path"])
        grid_search_param_grid = {key: tuple(value) for key, value in grid_search_param_grid.items()}
        bayes_search_cv_preprocess = BayesSearchCV(custom_estimator,
                                                   grid_search_param_grid,
                                                   cv=bayes_opt_search_config.getint("k_folds"),
                                                   n_iter=bayes_opt_search_config.getint("iterations"),
                                                   fit_params=self.fit_params)
        bayes_search_cv_preprocess.fit(self.X_data, self.y_data)
        return bayes_search_cv_preprocess


def get_grid_search_results(grid_search_cv_preprocess):
    grid_search_cv_results = grid_search_cv_preprocess.cv_results_
    for scores, parameters in zip(grid_search_cv_results["mean_test_score"], grid_search_cv_results["params"]):
        print(scores, parameters)


class TotalEstimator(BaseEstimator):

    def __init__(self,
                 missing_value_filter=0.5,
                 feature_importance_rank_filter=5,
                 n_estimators=200,
                 max_features=8,
                 max_leaf_nodes=25,
                 fit_params=None,
                 times_fit=1
                 ):
        self.missing_value_filter = missing_value_filter
        self.feature_importance_rank_filter = feature_importance_rank_filter
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.fit_params = fit_params
        self.times_fit = times_fit

    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        if list(self.fit_params.keys())[0] == "fit_params":
            col_na_proportion = self.fit_params["fit_params"]["col_na_proportion"]
        else:
            col_na_proportion = self.fit_params["col_na_proportion"]

        X, self.selected_features = feature_selection.feature_selection(
            X,
            y,
            col_na_proportion,
            self.missing_value_filter,
            self.feature_importance_rank_filter
        )

        X, y = check_X_y(X, y)

        if self.max_features > X.shape[1]:
            self.max_features = X.shape[1]

        self.baseline_rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            n_jobs=-1,
            oob_score=True
        )
        self.baseline_rf_model.fit(X, y)

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, y):
        X = X.loc[:, self.selected_features].to_numpy()
        check_is_fitted(self)
        X = check_array(X)
        return self.baseline_rf_model.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X, y)
        score = accuracy_score(y, y_pred)
        return score
