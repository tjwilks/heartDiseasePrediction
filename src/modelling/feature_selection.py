from sklearn.ensemble import RandomForestClassifier
import re

class FeatureSelector:
    """
        A class for searching selecting features based on
        missing value proportion or feature importance rank
        methods

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

        missing_value_filter : float
            threshold to above which columns that had a greater
            proportion of missing values before imputation (as
            is recorded in col_na_proportion) are removed

        feature_importance_rank_filter : int
            threshold bellow which columns with a lower
            feature importance rank are removed

        Methods
        -------
        select_on_missing_value_proportion
            removes columns that had a proportion of missing
            values before imputation (as is recorded in
            col_na_proportion) above threshold set by missing_value_filter

        select_on_feature_importance_rank
            removes columns that have a lower feature importance rank
            than threshold set by feature_importance_rank_filter
        """

    def __init__(self, X_data, y_data, col_na_proportion, missing_value_filter, feature_importance_rank_filter):
        self.X_data = X_data
        self.y_data = y_data
        self.col_na_proportion = col_na_proportion
        self.missing_value_filter = missing_value_filter
        self.feature_importance_rank_filter = feature_importance_rank_filter
        self.feature_importance_dict = None

    def select_on_missing_value_proportion(self):
        cols_na_proportion_above_filter = self.col_na_proportion[self.col_na_proportion > self.missing_value_filter]
        cols_with_x_prop_missing_values = cols_na_proportion_above_filter.index
        one_hot_cols = [col for col in self.X_data.columns if bool(re.search("_one_hot", col))]
        non_one_hot_cols = [col for col in self.X_data.columns if bool(re.search("_one_hot", col)) is False]
        cols_to_drop_one_hot = [col for col in one_hot_cols if re.split(r"_[^_]{0,20}_one_hot", col)[0] in cols_with_x_prop_missing_values]
        cols_to_drop_normal = [col for col in non_one_hot_cols if col in cols_with_x_prop_missing_values]
        cols_to_drop_total = list(set(cols_to_drop_normal).union(set(cols_to_drop_one_hot)))
        self.X_data = self.X_data.drop(cols_to_drop_total, axis=1)

    def select_on_feature_importance_rank(self):
        model = RandomForestClassifier(n_estimators=200, max_features=5, max_leaf_nodes=25, n_jobs=-1, oob_score=True)
        model.fit(self.X_data.to_numpy(), self.y_data)
        feature_importances = model.feature_importances_
        importance_rank = 1
        self.feature_importance_dict = {}
        for feature_importance, feature in sorted(zip(feature_importances, self.X_data.columns), reverse=True):
            self.feature_importance_dict[feature] = (importance_rank, feature_importance)
            importance_rank += 1
        self.selected_features = [feature for feature, rank_score in self.feature_importance_dict.items() if
                                  rank_score[0] <= self.feature_importance_rank_filter]
        self.X_data = self.X_data.loc[:, self.selected_features]
        self.X_data = self.X_data.to_numpy()
