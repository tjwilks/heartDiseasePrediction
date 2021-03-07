import os
import pandas as pd
import re
import numpy as np
import json


def get_data(heart_disease_data_directory, file_names):
    col_names = get_col_names("data/", "col_names.txt")
    data_list = []
    for file_name in file_names:
        raw_data = read_raw_data(heart_disease_data_directory, file_name)
        raw_data = process_raw_data(raw_data)
        data = generate_data_frame(raw_data, col_names)
        data["data_origin"] = file_name.replace(".data", "")
        data_list.append(data)
    all_data = pd.concat(data_list, axis=0)
    col_types = read_col_types("data/col_types.json")
    all_data = all_data.astype(col_types)
    return all_data


def get_col_names(heart_disease_data_directory,col_names_file_name):
    file_path = os.path.join(heart_disease_data_directory, col_names_file_name)
    with open(file_path, "rb") as file:
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


def read_col_types(col_types_json_file_path):
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
