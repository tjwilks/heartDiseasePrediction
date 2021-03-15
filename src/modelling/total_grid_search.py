from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from src.utils.preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from src.modelling import feature_selection
from sklearn.metrics import accuracy_score

def total_grid_search(
        X_data,
        y_data,
        missing_value_filter,
        feature_importance_rank_filter,
        max_leaf_nodes,
        max_features,
        n_estimators,
        fit_params):

    param_grid = {
        "missing_value_filter": missing_value_filter,
        "feature_importance_rank_filter": feature_importance_rank_filter,
        "max_leaf_nodes": max_leaf_nodes,
        "max_features": max_features,
        "n_estimators": n_estimators
    }
    custom_estimator = TotalEstimator()
    grid_search_cv_preprocess = GridSearchCV(custom_estimator, param_grid, cv=5)
    grid_search_cv_preprocess.fit(X_data, y_data, fit_params=fit_params)
    return grid_search_cv_preprocess


def get_grid_search_results(grid_search_cv_preprocess):
    grid_search_cv_results = grid_search_cv_preprocess.cv_results_
    for scores, parameters in zip(grid_search_cv_results["mean_test_score"], grid_search_cv_results["params"]):
        print(scores, parameters)


class TotalEstimator(BaseEstimator, DataPreprocessor):

    def __init__(self,
                 missing_value_filter=0.5,
                 feature_importance_rank_filter=5,
                 n_estimators=200,
                 max_features=8,
                 max_leaf_nodes=25,
                 ):
        self.missing_value_filter = missing_value_filter
        self.feature_importance_rank_filter = feature_importance_rank_filter
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y, fit_params):

        col_na_proportion = fit_params["col_na_proportion"]
        X, self.selected_features = feature_selection.feature_selection(
            X,
            y,
            col_na_proportion,
            self.missing_value_filter,
            self.feature_importance_rank_filter
        )

        X, y = check_X_y(X, y)

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
