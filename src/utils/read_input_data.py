import os
import pandas as pd
import re
import numpy as np
import json


def get_data(read_input_data_config):
    """
        function for loading data

        Parameters
        ----------
        read_input_data_config: configparser.SectionProxy
            configparser.SectionProxy generated from config.yml file
            specifying directory containing raw data files: heart_disease_data_dir
            data column names file path: col_names_file_path
            and file specifying data column types: col_types_file_path
    """

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
    return all_data


def get_col_names(col_names_file_path):
    """
        function for reading in data column names

        Parameters
        ----------
        col_names_file_path: str
            file path for txt file specifying column names

    """
    with open(col_names_file_path, "rb") as file:
        lines = file.readlines()
    col_names = [str(line)[2:-5] for line in lines]
    return col_names


def read_raw_data(heart_disease_data_directory, file_name):
    """
        function for reading raw data files

        Parameters
        ----------
        heart_disease_data_directory: str
            file path for txt file specifying column names

        file_name: str
            name of data file within heart_disease_data_directory
            to be read in as part of input data
    """
    file_path = os.path.join(heart_disease_data_directory, file_name)
    with open(file_path, "rb") as file:
        raw_data = file.read()
    raw_data = str(raw_data).rstrip(". name\\n\\n'")
    return raw_data


def process_raw_data(raw_data):
    """
        function for processing raw data stored as string into
        list of strings where each string in list represents
        a row of data

        Parameters
        ----------
        raw_data: str
            raw data to be processed
    """
    raw_data = str(raw_data)
    raw_data = raw_data.replace("b'", "")
    raw_data = raw_data.replace(". ", "")
    raw_data = re.sub("([^ ])(-9)", r'\1 \2', raw_data)
    raw_data = raw_data.replace("[^ ]-9", " -9")
    raw_data = raw_data.replace("\\n", " ")
    raw_data = raw_data.replace("'", "")
    raw_data = raw_data.split("name ")
    return raw_data


def read_json_to_dict(json_file_path):
    """
        function for reading in json files as dictionary

        Parameters
        ----------
        json_file_path: str
            file path for json file to be read in and converted to
            dictionary
    """
    with open(json_file_path, 'r') as j:
        json_as_dictionary = json.loads(j.read())
    return json_as_dictionary


def generate_data_frame(raw_data, col_names):
    """
        function for converting raw data as list of strings
        where each string in list represents a row of data into
        a pandas dataframe

        Parameters
        ----------
        raw_data: list
            list of strings where each string in list represents
            a row of data

        col_names: list
            list of strings where each string in list represents
            a row of data
    """
    rows = []
    for row in raw_data:
        row = row.rstrip(" ")
        row = row.split(" ")
        row = list(filter(lambda a: a != "", row))
        rows.append(row)
    data = pd.DataFrame(rows, columns=col_names)
    return data
