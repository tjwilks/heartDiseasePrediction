import unittest
from src.utils import read_input_data
import configparser
import pytest
import os


class TestReadInputData(unittest.TestCase):
    """
        a class for testing the read input data module

        Methods
        -------
        setup_test:
            initialises read_input_data variables used in multiple tests

        test_row_lengths_equal:
            tests row lengths of raw data are all equal

        test_n_col_names_equal_to_n_cols:
            tests number of column names specified in col_names_file_path
            file is equal to number of columns

        test_n_col_types_equal_to_n_cols:
            tests number of column names specified in col_types_file_path
            file is equal to number of columns

        test_col_types_float_or_str:
            tests col types specified in col_types_file_path file are
            either 'string' or 'float'

        test_n_rows:
            tests number of rows is not 0 and greater than number of columns
    """


    @pytest.fixture(autouse=True)
    def setup_test(self):
        self.config = configparser.ConfigParser()
        self.config.read("config/config.yml")
        self.config = self.config["read_input_data_config"]
        self.data = read_input_data.get_data(self.config)
        self.col_names = read_input_data.get_col_names(self.config["col_names_file_path"])
        self.col_names.append("data_origin")
        self.col_types = read_input_data.read_json_to_dict(self.config["col_types_file_path"])

    def test_row_lengths_equal(self):
        file_names = os.listdir(self.config["heart_disease_data_dir"])
        row_lengths = []
        for file_name in file_names:
            raw_data = read_input_data.read_raw_data(self.config["heart_disease_data_dir"], file_name)
            raw_data = read_input_data.process_raw_data(raw_data)
            for row in raw_data:
                row = row.rstrip(" ")
                row = row.split(" ")
                row = list(filter(lambda a: a != "", row))
                row_length = len(row)
                row_lengths.append(row_length)
        unique_row_lengths = set(row_lengths)
        assert len(unique_row_lengths) == 1

    def test_n_col_names_equal_to_n_cols(self):
        n_col_names = len(self.col_names)
        n_cols = self.data.shape[1]
        assert n_col_names == n_cols

    def test_n_col_types_equal_to_n_cols(self):
        n_col_types = len(self.col_types)
        n_cols = self.data.shape[1]
        assert n_col_types == n_cols

    def test_col_types_float_or_str(self):
        col_types_dict_values = self.col_types.values()
        unique_col_types = set(col_types_dict_values)
        expected_col_types = set(["float", "string"])
        col_types_set_diff = unique_col_types - expected_col_types
        assert len(col_types_set_diff) == 0

    def test_n_rows(self):
        n_rows = self.data.shape[0]
        n_cols = self.data.shape[1]
        assert n_rows > n_cols
        assert n_rows != 0


