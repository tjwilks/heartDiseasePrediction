from src.utils import read_input_data
from src.utils.preprocessing import DataPreprocessor
import configparser
import pandas as pd
import numpy as np
import pytest
import re

class TestPreprocessing:

    """
        a class for testing preprocessing

        Methods
        -------
        setup_test:
            initialises DataPreprocessor object

        test_imputation:
            tests all missing values are imputed

        test_scaling:
            tests all values are between 0 and 1

        test_only_binary_cat_variables:
            tests all missing values are imputed
    """

    @pytest.fixture(autouse=True)
    def setup_test(self):
        # read in package config
        config = configparser.ConfigParser()
        config.read("config/config.yml")

        # read in input data
        all_data = read_input_data.get_data(
            config["read_input_data_config"]
        )
        # preprocess input data
        self.data_preprocessor = DataPreprocessor(all_data)
        self.data_preprocessor.preprocess_data()

    def test_imputation(self):
        total_nan = np.array(self.data_preprocessor.X_data == np.nan).sum()
        total_null = np.array(self.data_preprocessor.X_data.isnull()).sum()
        total_na = np.array(self.data_preprocessor.X_data.isna()).sum()
        total = total_nan + total_null + total_na
        assert total == 0

    def test_scaling(self):
        X_data_as_float = self.data_preprocessor.X_data.astype("float")
        assert all(X_data_as_float.max() == 1)
        assert all(X_data_as_float.min() == 0)

    def test_only_binary_cat_variables(self):
        cat_data_nunique_vals = self.data_preprocessor.cat_data.nunique().values
        assert all(cat_data_nunique_vals == 2)
