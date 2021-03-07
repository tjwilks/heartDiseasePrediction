import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class FeatureSelector:

    def __init__(self, data, missing_value_filter):
        self.data = data
        self.missing_value_filter = missing_value_filter

    def drop_unessersary_cols(self):
        # judged as unessersary by examination of column names
        col_to_drop_idxs = [0, 1]
        cols_to_drop = self.data.columns[col_to_drop_idxs]
        self.data = self.data.drop(cols_to_drop, axis=1)
        return self.data

    def drop_cols_with_x_prop_missing_values(self):
        self.data = self.data.replace({pd.NA: np.nan})
        col_na_proportion = self.data.isnull().sum() / len(self.data)
        col_na_proportion = col_na_proportion[col_na_proportion < self.missing_value_filter]
        cols_to_keep = col_na_proportion.index
        self.data = self.data.loc[:, cols_to_keep]


class ColAssignor:

    def __init__(self, data):
        self.data = data
        self.cat_data = None
        self.cont_data = None
        self.y_data = None

    def assign_output_col(self):
        y_col_name = "58 num: diagnosis of heart disease (angiographic disease status)"
        self.y_data = self.data[y_col_name]
        self.data = self.data.drop(y_col_name, axis =1)

    def assign_col_types(self):
        continuous_col_type_idxs = [2, 9, 11, 13, 14, 15, 28, 29, 30, 31,
                                    32, 33, 34, 35, 36, 39, 41, 42, 43, 44]
        col_names = self.data.columns
        i = 0
        dtypes_dict = {}
        for col_name in col_names:
            if i in continuous_col_type_idxs:
                dtypes_dict[col_name] = 'float'
            else:
                dtypes_dict[col_name] = 'string'
            i += 1
        self.data = self.data.astype(dtypes_dict)
        self.cat_data = self.data.select_dtypes("string")
        self.cont_data = self.data.select_dtypes("float")


class OutputTransformer:

    def __init__(self, y_data):
        self.y_data = y_data

    def recategorise_output_categories(self):
        self.y_data = np.where(self.y_data == "0", 0, 1)


class CategoricalVariablePreprocessor:

    def __init__(self, data):
        self.data = data
        self.cat_data = None

    def impute_cat_missing_values(self):
        self.cat_data = self.data.select_dtypes("string")
        col_names = self.cat_data.columns
        self.cat_data = self.cat_data.replace({pd.NA: np.nan})
        most_frequent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.cat_data = most_frequent_imputer.fit_transform(self.cat_data.to_numpy())
        self.cat_data = pd.DataFrame(self.cat_data, columns=col_names)

    def one_hot_encode_cat_variables(self):
        one_hot_encoder = OneHotEncoder()
        cat_data_one_hot = one_hot_encoder.fit_transform(self.cat_data)
        one_hot_features = one_hot_encoder.get_feature_names(list(self.cat_data.columns))
        cat_data_one_hot = cat_data_one_hot.toarray()
        self.cat_data = pd.DataFrame(cat_data_one_hot, columns=one_hot_features)


class ContinuousVariablePreprocessor:

    def __init__(self, data):
        self.data = data
        self.cont_data = None

    def impute_cont_missing_values(self):
        self.cont_data = self.data.select_dtypes("float")
        self.cont_data = self.cont_data.replace({pd.NA: np.nan})
        col_names = self.cont_data.columns
        median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.cont_data = median_imputer.fit_transform(self.cont_data)
        self.cont_data = pd.DataFrame(self.cont_data, columns=col_names)

    def normalise_continuous_variables(self):
        col_names = self.data.columns
        for col_name in col_names:
            if self.data[col_name].dtype == "float":
                self.data[col_name] = (self.data[col_name] - np.min(self.data[col_name])) / (
                        np.max(self.data[col_name]) - np.min(self.data[col_name]))


class InputDataFormatting:

    def __init__(self, cat_data, cont_data):
        self.cat_data = cat_data
        self.cont_data = cont_data
        self.features_df = None
        self.features = None
        self.X_data = None

    def concatenate_cat_cont_data(self):
        self.features_df = pd.concat([self.cat_data, self.cont_data], axis=1)
        self.features = self.features_df.columns

    def prepare_data_for_modelling(self):
        self.X_data = self.features_df.to_numpy()


class DataPreprocessor(FeatureSelector,
                       ColAssignor,
                       OutputTransformer,
                       CategoricalVariablePreprocessor,
                       ContinuousVariablePreprocessor,
                       InputDataFormatting):

    def __init__(self, data, missing_value_filter):
        self.data = data
        self.missing_value_filter = missing_value_filter

    def preprocess_data(self):

        self.label_missing_values()
        # col removal
        self.drop_unessersary_cols()
        self.drop_cols_with_x_prop_missing_values()
        # col assignment
        self.assign_output_col()
        self.assign_col_types()
        # output transformation
        self.recategorise_output_categories()
        # categorical variable preprocessing
        self.impute_cat_missing_values()
        self.one_hot_encode_cat_variables()
        # continuous variable preprocessing
        self.normalise_continuous_variables()
        self.impute_cont_missing_values()
        # input data formatting
        self.concatenate_cat_cont_data()
        self.prepare_data_for_modelling()


    def label_missing_values(self):
        col_types = self.data.dtypes
        self.data = self.data.replace({'-9': np.nan, -9: np.nan})
        self.data = self.data.astype(col_types)

