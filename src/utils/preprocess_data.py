import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(data):
    data = assign_catagorical_col_types(data)
    data = drop_unessersary_cols(data)
    data = drop_cols_with_x_prop_missing_values(data, 0.2)
    data = order_y_output(data)
    cat_data = impute_cat_missing_values(data)
    cat_data = one_hot_encode_cat_variables(cat_data)
    cont_data = impute_cont_missing_values(data)
    cont_data = normalise_continuous_variables(cont_data)
    y_data = data["y"]
    X_data = pd.concat([cat_data, cont_data], axis=1)
    return X_data, y_data


def assign_catagorical_col_types(data):
    continuous_col_type_idxs = [2, 9, 11, 13, 14, 15, 28, 29, 30,
                                31, 32, 33, 34, 35, 36, 39, 41, 42, 43, 44]
    col_names = data.columns
    i = 0
    dtypes_dict = {}
    for col_name in col_names:
        if i in continuous_col_type_idxs:
            dtypes_dict[col_name] = 'float'
        else:
            dtypes_dict[col_name] = 'string'
        i += 1
    data = data.astype(dtypes_dict)
    return data


def drop_unessersary_cols(data):
    # judged as unessersary by examination of column names
    col_to_drop_idxs = [0, 1, 19, 20, 21, 22, 27, 35, 44, 45, 51, 52, 53, 54, 55, 56, 68, 69, 70, 71, 72, 73, 74]
    cols_to_drop = data.columns[col_to_drop_idxs]
    data = data.drop(cols_to_drop, axis=1)
    return data


def drop_cols_with_x_prop_missing_values(data, x_prop):
    col_na_proportion = data.isnull().sum() / len(data)
    col_na_proportion = col_na_proportion[col_na_proportion < x_prop]
    cols_to_keep = col_na_proportion.index
    data = data.loc[:, cols_to_keep]
    return data


def normalise_continuous_variables(data):
    col_names = data.columns
    for col_name in col_names:
        if data[col_name].dtype == "float":
            data[col_name] = (data[col_name] - np.min(data[col_name]))/(np.max(data[col_name]) - np.min(data[col_name]))
    return data


def order_y_output(data):
    data["y"] = data["58 num: diagnosis of heart disease (angiographic disease status)"]
    data = data.drop("58 num: diagnosis of heart disease (angiographic disease status)", axis=1)
    return data


def impute_cat_missing_values(cat_data):
    cat_data = cat_data.select_dtypes("string")
    cat_data = cat_data.replace({pd.NA: np.nan})
    col_names = cat_data.columns
    most_frequent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cat_data = most_frequent_imputer.fit_transform(cat_data.to_numpy())
    cat_data = pd.DataFrame(cat_data, columns=col_names)
    return cat_data


def impute_cont_missing_values(cont_data):
    cont_data = cont_data.select_dtypes("float")
    col_names = cont_data.columns
    median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    cont_data = median_imputer.fit_transform(cont_data)
    cont_data = pd.DataFrame(cont_data, columns=col_names)
    return cont_data


def one_hot_encode_cat_variables(cat_data):
    cat_data = cat_data.drop("y", axis=1)
    one_hot_encoder = OneHotEncoder()
    cat_data = one_hot_encoder.fit_transform(cat_data)
    cat_data = cat_data.toarray()
    cat_data = pd.DataFrame(cat_data)
    return cat_data
