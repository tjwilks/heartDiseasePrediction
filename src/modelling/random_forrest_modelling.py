from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_random_forrests(X_data, y_data):
    baseline_rf_model = RandomForestClassifier(n_estimators=200, max_features=5, max_leaf_nodes=25, n_jobs=-1, oob_score=True)
    baseline_rf_model.fit(X_data, y_data)
    out_of_bag_score = baseline_rf_model.oob_score_
    cross_validation_score = np.mean(cross_val_score(baseline_rf_model, X_data, y_data, cv=10))
    print(f"Out of bag score: {str(out_of_bag_score)}")
    print(f"cross validation score: {str(cross_validation_score)}")
    return baseline_rf_model


def random_forrest_grid_search(X_data, y_data):
    param_grid = {
        "max_leaf_nodes": [10, 15, 20, 25, 30],
        "max_features": [3, 4, 5, 6, 7]
    }
    forrest_classifier = RandomForestClassifier(n_estimators=200)
    grid_search_cv_rf = GridSearchCV(forrest_classifier, param_grid, cv=10)
    grid_search_cv_rf.fit(X_data, y_data)
    return grid_search_cv_rf


def grid_search_cv_rf_results(grid_search_cv_rf):
    grid_search_cv_results = grid_search_cv_rf.cv_results_
    for scores, parameters in zip(grid_search_cv_results["mean_test_score"], grid_search_cv_results["params"]):
        print(scores, parameters)


def random_forrest_feature_importances(model, features):
    feature_importances = model.feature_importances_
    importance_rank = 1
    feature_importance_dict = {}
    for feature_importance, feature in sorted(zip(feature_importances, features), reverse=True):
        print(f"{str(importance_rank)}: {feature[2:]}: {np.round(float(feature_importance), 3)}")
        feature_importance_dict[feature] = (importance_rank, feature_importance)
        importance_rank += 1
    return feature_importance_dict
