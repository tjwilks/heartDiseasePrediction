from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from src.modelling.feature_selection import FeatureSelector
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
import numpy as np
from src.utils import read_input_data


class HypParamSearch:
    """
        A class for searching for optimal hyperparameters using
        either grid search or bayesian optimisation

        Attributes
        ----------
        X_data : pandas dataframe
            pandas dataframe containing data to be used for training
            on and predicting output variable y_data

        y_data : numpy array
            numpy array containing output variable to be predicted by
            X_data

        col_na_proportion : dict
            dictionary containing the proportion of missing values
            present in each variable within X_data for the purpose of
            missing value based feature selection

        Methods
        -------
        grid_hyp_search(grid_search_config=dict)
            conducts grid search with across hyperparameter range
            specified in grid_search_config (as controlled by
            grid_search_param_grid.json)

        bayes_opt_hyp_search(bayes_opt_search_config=dict)
            conducts bayesian optimisation hyperparameter search
            across range specified in bayes_opt_search_config (as
            controlled by bayes_opt_search_param_grid.json)
        """

    def __init__(self, X_data, y_data, col_na_proportion):
        self.X_data = X_data
        self.y_data = y_data
        self.fit_params = {"col_na_proportion": col_na_proportion}

    def grid_hyp_search(self, grid_search_config):
        """
        Parameters
        ----------
        grid_search_config : dict
            conducts grid search with across hyperparameter range
            specified in grid_search_config (as controlled by
            grid_search_param_grid.json)
        """
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
        """
        Parameters
        ----------
        bayes_opt_search_config : dict
            conducts bayesian optimisation hyperparameter search
            across range specified in bayes_opt_search_config (as
            controlled by bayes_opt_search_param_grid.json)
        """
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
    """
        function for returning hyperparameter search results to
        console

        Parameters
        ----------
        grid_search_cv_preprocess: object
            fitted GridSearchCV or BayesSearchCV hyperparameter
            optimisation object
    """
    grid_search_cv_results = grid_search_cv_preprocess.cv_results_
    for scores, parameters in zip(grid_search_cv_results["mean_test_score"], grid_search_cv_results["params"]):
        print(f"Cross-val error: {np.round(scores, 5)} --- Parameters: {parameters}")
    print("-------------------------")
    print(f"Best score found: {np.round(grid_search_cv_preprocess.best_score_, 5)}")
    print(f"Best hyperparameter configuration found: {grid_search_cv_preprocess.best_params_}")


class TotalEstimator(BaseEstimator):
    """
        A custom scikit-learn estimator that varies preprocessing
        and modelling based on specified hyperparameters
        and produces associated predictions and error scores

        Attributes
        ----------
        missing_value_filter : float
            threshold to above which columns that had a greater
            proportion of missing values before imputation (as
            is recorded in col_na_proportion) are removed

        feature_importance_rank_filter : int
            threshold bellow which columns with a lower
            feature importance rank are removed

        n_estimators : int
            The number of trees in the forest used in random
            forrest model

        max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        fit_params : dict
            Parameters to pass to the fit method.

        times_fit: int
            integer for tracking the progress of hyperparameter search
            by counting number of times fit method is called

        Methods
        -------
        fit(X=pandas dataframe, y=numpy array, **fit_params=dict)
            fit estimator

        predict(X=pandas dataframe)
            predict based on fitted estimator

        score(X=pandas dataframe, y=numpy array)
            calculate accuracy score of estimator predictions
        """

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
        """
        Parameters
        ----------
        X : pandas dataframe
            pandas dataframe containing data to be used for model
            fitting

        y : numpy array
            numpy array containing output variable to be used for
            model fitting

        fit_params : dict
            dictionary containing additional parameters to be used in
            model fitting.
        """
        # handle fit params depending on whether estimator attribute or
        # fit method parameter
        self.fit_params = fit_params
        if list(self.fit_params.keys())[0] == "fit_params":
            col_na_proportion = self.fit_params["fit_params"]["col_na_proportion"]
        else:
            col_na_proportion = self.fit_params["col_na_proportion"]

        feature_selector = FeatureSelector(X,
                                           y,
                                           col_na_proportion,
                                           self.missing_value_filter,
                                           self.feature_importance_rank_filter)
        feature_selector.select_on_missing_value_proportion()
        feature_selector.select_on_feature_importance_rank()
        X = feature_selector.X_data
        self.selected_features = feature_selector.selected_features

        # if feature selection has removed columns so that number
        # of columns remaining is less than max features
        # set max features to be number of columns remaining
        if self.max_features > X.shape[1]:
            self.max_features = X.shape[1]

        # Random forrest classification
        self.baseline_rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            n_jobs=-1,
            oob_score=True
        )
        self.baseline_rf_model.fit(X, y)

        # required functionality for sci-kit learn estimator
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : pandas dataframe
            pandas dataframe containing data to be used
            predict output y variable based on fitted
            estimator model
        """
        X = X.loc[:, self.selected_features].to_numpy()
        check_is_fitted(self)
        X = check_array(X)
        return self.baseline_rf_model.predict(X)

    def score(self, X, y):
        """
        Parameters
        ----------
        X : pandas dataframe
            pandas dataframe containing data to be used
            predict output y variable based on fitted
            estimator model

        y : numpy array
            output variable to be used to compare against
            estimator predictions for score to be calculated
        """
        y_pred = self.predict(X)
        score = accuracy_score(y, y_pred)
        return score
