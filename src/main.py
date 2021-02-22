import pandas as pd
pd.set_option('display.max_columns',40)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)

import configparser
from src.utils import read_input_data, preprocess_data


def main():
    config = configparser.ConfigParser()
    config.read("config.yml")
    data = read_input_data.get_data(
        config["default"].get("heart_disease_data_dir"),
        ["switzerland.data",
         "long-beach-va.data",
         "hungarian.data"]
    )
    X_data, y_data = preprocess_data.preprocess_data(data)

if __name__ == "__main__":
    main()
