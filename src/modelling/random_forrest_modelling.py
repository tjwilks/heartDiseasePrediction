from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_random_forrests(X_data, y_data):
    baseline_rf_model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, oob_score=True)
    baseline_rf_model.fit(X_data, y_data)
    out_of_bag_score = baseline_rf_model.oob_score_
    cross_validation_score = np.mean(cross_val_score(baseline_rf_model, X_data, y_data, cv=10))
    print(f"Out of bag score: {str(out_of_bag_score)}")
    print(f"cross validation score: {str(cross_validation_score)}")


