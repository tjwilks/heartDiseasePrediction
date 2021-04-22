import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class ColAssignor:

    def __init__(self, X_data):
        self.X_data = X_data
        self.cat_data = None
        self.cont_data = None
        self.y_data = None

    def assign_output_col(self):
        y_col_name = "58 num: diagnosis of heart disease (angiographic disease status)"
        self.y_data = self.X_data[y_col_name]
        self.X_data = self.X_data.drop(y_col_name, axis =1)

    def assign_col_types(self):
        self.cat_data = self.X_data.select_dtypes("string")
        self.cont_data = self.X_data.select_dtypes("float")


class CategoricalVariablePreprocessor:

    def __init__(self, X_data):
        self.X_data = X_data
        self.cat_data = None

    def impute_cat_missing_values(self):
        self.cat_data = self.X_data.select_dtypes("string")
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

    def __init__(self, X_data):
        self.X_data = X_data
        self.cont_data = None

    def impute_cont_missing_values(self):
        self.cont_data = self.X_data.select_dtypes("float")
        self.cont_data = self.cont_data.replace({pd.NA: np.nan})
        col_names = self.cont_data.columns
        median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.cont_data = median_imputer.fit_transform(self.cont_data)
        self.cont_data = pd.DataFrame(self.cont_data, columns=col_names)

    def normalise_continuous_variables(self):
        col_names = self.cont_data.columns
        for col_name in col_names:
            col = self.cont_data.loc[:, col_name]
            if np.max(col) - np.min(col) != 0:
                min = np.min(col)
                min_max_diff = np.max(col) - min
                self.cont_data.loc[:, col_name] = (col - min) / min_max_diff


class InputDataFormatting:

    def __init__(self, cat_data, cont_data):
        self.cat_data = cat_data
        self.cont_data = cont_data
        self.features_df = None
        self.features = None
        self.X_data = None

    def concatenate_cat_cont_data(self):
        self.X_data = pd.concat([self.cat_data, self.cont_data], axis=1)
        self.features = self.X_data.columns


class DataPreprocessor(ColAssignor,
                       CategoricalVariablePreprocessor,
                       ContinuousVariablePreprocessor,
                       InputDataFormatting):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def preprocess_data(self):

        self.label_missing_values()
        self.caluculate_na_proportions()

        self.remove_all_na_cols()
        self.drop_unessersary_cols()
        self.assign_col_types()

        self.impute_cont_missing_values()
        self.normalise_continuous_variables()
        # categorical variable preprocessing
        self.impute_cat_missing_values()
        self.one_hot_encode_cat_variables()
        # continuous variable preprocessing
        self.normalise_continuous_variables()
        self.impute_cont_missing_values()
        # input data formatting
        self.concatenate_cat_cont_data()

    def drop_unessersary_cols(self):
        # judged as unessersary by examination of column names
        cols_to_drop = ['1 id: patient identification number',
                        '2 ccf: social security number (I replaced this with a dummy value of 0)',
                        '20 ekgmo (month of exercise ECG reading)',
                        '21 ekgday(day of exercise ECG reading)',
                        '22 ekgyr (year of exercise ECG reading)',
                        '55 cmo: month of cardiac cath (sp?)  (perhaps "call")',
                        '56 cday: day of cardiac cath (sp?)',
                        '57 cyr: year of cardiac cath (sp?)']

        self.X_data = self.X_data.drop(cols_to_drop, axis=1)
        return self.X_data

    def label_missing_values(self):
        col_types = self.X_data.dtypes
        self.X_data = self.X_data.replace({'-9': np.nan, -9: np.nan})
        self.X_data = self.X_data.astype(col_types)

    def caluculate_na_proportions(self):
        X_data_np_nan = self.X_data.replace({pd.NA: np.nan})
        self.col_na_proportion = X_data_np_nan.isnull().sum() / len(X_data_np_nan)

    def remove_all_na_cols(self):
        all_na_cols = self.col_na_proportion[self.col_na_proportion >= 0.98]
        all_na_cols = all_na_cols.index
        self.X_data = self.X_data.drop(all_na_cols, axis=1)

    def assign_col_types(self):
        self.cat_data = self.X_data.select_dtypes("string")
        self.cont_data = self.X_data.select_dtypes("float")



