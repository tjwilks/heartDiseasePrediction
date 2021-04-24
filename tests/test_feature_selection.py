from src.utils import read_input_data
from src.utils.preprocessing import DataPreprocessor
from src.modelling.feature_selection import FeatureSelector
import configparser
import pytest
import re

class TestFeatureSelection:
    """
        a class for testing feature selection module

        Methods
        -------
        setup_test:
            initialises FeatureSelector object

        test_cols_dropped_by_missing_value_prop:
            tests select_on_missing_value_proportion removes columns

        test_cols_dropped_by_feature_importance_rank:
            tests select_on_feature_importance_rank removes columns

        test_no_cols_after_na_selection_above_threshold:
            tests columns remaining after missing value feature selection
            applied have na proportion above threshold

    """

    @pytest.fixture(autouse=True)
    def setup_test(self):
        self.config = configparser.ConfigParser()
        self.config.read("config/config.yml")
        self.config = self.config["read_input_data_config"]
        self.data = read_input_data.get_data(self.config)
        data_preprocessor = DataPreprocessor(self.data)
        data_preprocessor.preprocess_data()
        self.feature_selector = FeatureSelector(X_data=data_preprocessor.X_data,
                                                y_data=data_preprocessor.y_data,
                                                col_na_proportion=data_preprocessor.col_na_proportion,
                                                missing_value_filter=0.5,
                                                feature_importance_rank_filter=20)

    def test_cols_dropped_by_missing_value_prop(self):
        n_cols_before = self.feature_selector.X_data.shape[1]
        self.feature_selector.select_on_missing_value_proportion()
        n_cols_after = self.feature_selector.X_data.shape[1]
        assert n_cols_before > n_cols_after

    def test_cols_dropped_by_feature_importance_rank(self):
        n_cols_before = self.feature_selector.X_data.shape[1]
        self.feature_selector.select_on_feature_importance_rank()
        n_cols_after = self.feature_selector.X_data.shape[1]
        assert n_cols_before > n_cols_after

    def test_no_cols_after_na_selection_above_threshold(self):
        self.feature_selector.select_on_missing_value_proportion()
        columns_remaining = self.feature_selector.X_data.columns
        original_columns_remaining_one_hot = [re.split(r"_[^_]{0,20}_one_hot", col)[0] for col in columns_remaining
                                              if bool(re.search(r"_[^_]{0,20}_one_hot", col))]
        columns_remaining_non_one_hot = [col for col in columns_remaining
                                         if bool(re.search(r"_[^_]{0,20}_one_hot", col)) == False]
        original_columns_remaining = set(original_columns_remaining_one_hot).union(set(columns_remaining_non_one_hot))

        cols_na_prop_bellow = self.feature_selector.col_na_proportion[self.feature_selector.col_na_proportion <=
                                                                      self.feature_selector.missing_value_filter]
        cols_na_prop_bellow = cols_na_prop_bellow.index
        na_bellow_not_remaining = set(original_columns_remaining) - set(cols_na_prop_bellow)
        n_cols_na_bellow_not_remaining = len(na_bellow_not_remaining)
        assert n_cols_na_bellow_not_remaining == 0
