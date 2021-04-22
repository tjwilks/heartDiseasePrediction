import os
import pandas as pd
import re
import numpy as np
import json


def get_data(read_input_data_config):
    col_names = get_col_names(read_input_data_config["col_names_file_path"])
    file_names = os.listdir(read_input_data_config["heart_disease_data_dir"])
    data_list = []
    for file_name in file_names:
        raw_data = read_raw_data(read_input_data_config["heart_disease_data_dir"], file_name)
        raw_data = process_raw_data(raw_data)
        data = generate_data_frame(raw_data, col_names)
        data["data_origin"] = file_name.replace(".data", "")
        data_list.append(data)
    all_data = pd.concat(data_list, axis=0)
    col_types = read_json_to_dict(read_input_data_config["col_types_file_path"])
    all_data = all_data.astype(col_types)
    X_data, y_data = assign_output_col(all_data)
    y_data = recategorise_output_categories(y_data)
    return X_data, y_data


def get_col_names(col_names_file_path):
    with open(col_names_file_path, "rb") as file:
        lines = file.readlines()
    col_names = [str(line)[2:-5] for line in lines]
    return col_names


def read_raw_data(heart_disease_data_directory, file_name):
    file_path = os.path.join(heart_disease_data_directory, file_name)
    with open(file_path, "rb") as file:
        raw_data = file.read()
    raw_data = str(raw_data).rstrip(". name\\n\\n'")
    return raw_data


def process_raw_data(raw_data):
    raw_data = str(raw_data)
    raw_data = raw_data.replace("b'", "")
    raw_data = raw_data.replace(". ", "")
    raw_data = re.sub("([^ ])(-9)", r'\1 \2', raw_data)
    raw_data = raw_data.replace("[^ ]-9", " -9")
    raw_data = raw_data.replace("\\n", " ")
    raw_data = raw_data.replace("'", "")
    raw_data = raw_data.split("name ")
    return raw_data


def read_json_to_dict(col_types_json_file_path):
    with open(col_types_json_file_path, 'r') as j:
        col_types = json.loads(j.read())
    return col_types


def generate_data_frame(raw_data, col_names):
    rows = []
    for row in raw_data:
        row = row.rstrip(" ")
        row = row.split(" ")
        row = list(filter(lambda a: a != "", row))
        rows.append(row)
    data = pd.DataFrame(rows, columns=col_names)
    return data


def assign_output_col(data):
    y_col_name = "58 num: diagnosis of heart disease (angiographic disease status)"
    y_data = data[y_col_name]
    X_data = data.drop(y_col_name, axis =1)
    return X_data, y_data


def recategorise_output_categories(y_data):
    y_data = np.where(y_data == "0", 0, 1)
    return y_data