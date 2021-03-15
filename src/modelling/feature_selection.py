from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.modelling.preprocessing_estimator import CustomEstimator


def feature_selection_grid_search(X_data, y_data, fit_params):
    param_grid = {
        "missing_value_filter": [0.5, 0.8],
        "feature_importance_rank_filter": [8, 12, 17]
    }
    custom_estimator = CustomEstimator()
    grid_search_cv_preprocess = GridSearchCV(custom_estimator, param_grid, cv=5)
    grid_search_cv_preprocess.fit(X_data, y_data, fit_params=fit_params)
    return grid_search_cv_preprocess


def get_feature_selection_grid_search_results(grid_search_cv_preprocess):
    grid_search_cv_results = grid_search_cv_preprocess.cv_results_
    for scores, parameters in zip(grid_search_cv_results["mean_test_score"], grid_search_cv_results["params"]):
        print(scores, parameters)


def feature_selection(X_data,
                      y_data,
                      col_na_proportion,
                      missing_value_filter,
                      feature_importance_rank_filter):

    feature_selector = FeatureSelector(X_data, y_data, col_na_proportion, missing_value_filter, feature_importance_rank_filter)
    feature_selector.drop_cols_with_x_prop_missing_values()
    feature_selector.random_forrest_feature_importances()
    feature_selector.feature_importance_feature_selection()
    feature_selector.filter_X_data_to_selected_features()

    return feature_selector.X_data, feature_selector.selected_features


class FeatureSelector:

    def __init__(self, X_data, y_data, col_na_proportion, missing_value_filter, feature_importance_rank_filter):
        self.X_data = X_data
        self.y_data = y_data
        self.col_na_proportion = col_na_proportion
        self.missing_value_filter = missing_value_filter
        self.feature_importance_rank_filter = feature_importance_rank_filter
        self.feature_importance_dict = None

    def drop_cols_with_x_prop_missing_values(self):
        cols_na_proportion_above_filter = self.col_na_proportion[self.col_na_proportion > self.missing_value_filter]
        cols_with_x_prop_missing_values = cols_na_proportion_above_filter.index
        cols_to_drop = set(cols_with_x_prop_missing_values).intersection(set(self.X_data.columns))
        self.X_data = self.X_data.drop(cols_to_drop, axis=1)

    def random_forrest_feature_importances(self):
        model = RandomForestClassifier(n_estimators=200, max_features=5, max_leaf_nodes=25, n_jobs=-1, oob_score=True)
        model.fit(self.X_data.to_numpy(), self.y_data)
        feature_importances = model.feature_importances_
        importance_rank = 1
        self.feature_importance_dict = {}
        for feature_importance, feature in sorted(zip(feature_importances, self.X_data.columns), reverse=True):
            self.feature_importance_dict[feature] = (importance_rank, feature_importance)
            importance_rank += 1

    def feature_importance_feature_selection(self):
        self.selected_features = [feature for feature, rank_score in self.feature_importance_dict.items() if
                                  rank_score[0] <= self.feature_importance_rank_filter]

    def filter_X_data_to_selected_features(self):
        self.X_data = self.X_data.loc[:, self.selected_features]
        self.X_data = self.X_data.to_numpy()
