import configparser
from src.utils import read_input_data, preprocess_data
from src.modelling import random_forrest_modelling


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
    model = random_forrest_modelling.train_random_forrests(X_data, y_data)


if __name__ == "__main__":
    main()

