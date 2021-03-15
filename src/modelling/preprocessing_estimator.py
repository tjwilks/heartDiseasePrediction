import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from src.utils.preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from src.modelling import random_forrest_modelling, feature_selection


class CustomEstimator(BaseEstimator, DataPreprocessor):

    def __init__(self, missing_value_filter=0.5, feature_importance_rank_filter=5):
        self.missing_value_filter = missing_value_filter
        self.feature_importance_rank_filter = feature_importance_rank_filter

    def fit(self,
            X,
            y,
            fit_params):

        col_na_proportion = fit_params["col_na_proportion"]
        X, self.selected_features = feature_selection.feature_selection(
            X,
            y,
            col_na_proportion,
            self.missing_value_filter,
            self.feature_importance_rank_filter
        )

        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.baseline_rf_model = RandomForestClassifier(n_estimators=200,
                                                        max_features=8,
                                                        max_leaf_nodes=25,
                                                        n_jobs=-1,
                                                        oob_score=True
                                                        )

        self.baseline_rf_model.fit(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X, y):

        X = X.loc[:, self.selected_features].to_numpy()
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        return self.baseline_rf_model.predict(X)

    def score(self, X, y):
        # counts number of values bigger than mean
        y_pred = self.predict(X, y)
        score = sum(y_pred==y)/len(y_pred)
        return score