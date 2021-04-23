import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class CategoricalVariablePreprocessor:
    """
         A class for preprocessing categorical data

         Attributes
         ----------
         cat_data : pandas dataframe
             pandas dataframe containing categorical data with
             string col types to be preprocessed

         Methods
         -------
         impute_cat_missing_values
             impute categorical variables with most common column category

         one_hot_encode_cat_variables
             transform categorical variables to binary categorical variables
     """

    def __init__(self, cat_data):
        self.cat_data = cat_data

    def impute_cat_missing_values(self):
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
    """
        A class for preprocessing continuous data

        Attributes
        ----------
        cont_data : pandas dataframe
            pandas dataframe containing continuous data with
            float col types to be preprocessed

        Methods
        -------
        impute_cont_missing_values
            impute continuous variables with median value of column

        normalise_cont_data
            normalise continuous variables to be between 0 and 1
    """
    def __init__(self, cont_data):
        self.cont_data = cont_data

    def impute_cont_missing_values(self):
        col_names = self.cont_data.columns
        self.cont_data = self.cont_data.replace({pd.NA: np.nan})
        median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        self.cont_data = median_imputer.fit_transform(self.cont_data)
        self.cont_data = pd.DataFrame(self.cont_data, columns=col_names)

    def normalise_cont_data(self):
        col_names = self.cont_data.columns
        for col_name in col_names:
            col = self.cont_data.loc[:, col_name]
            if np.max(col) - np.min(col) != 0:
                min = np.min(col)
                min_max_diff = np.max(col) - min
                self.cont_data.loc[:, col_name] = (col - min) / min_max_diff


class DataPreprocessor(CategoricalVariablePreprocessor,
                       ContinuousVariablePreprocessor):
    """
        A class for preprocessing data

        Attributes
        ----------
        X_data : pandas dataframe
            pandas dataframe containing data to be used for training
            on and predicting output variable y_data

        y_data : numpy array
            numpy array containing output variable to be predicted by
            X_data

        Methods
        -------
        label_missing_values
            label inputted value as a np.nan

        calculate_na_proportions
            calculate the proportion of missing values for each column
            of inputted data

        remove_all_na_cols
            remove columns that are greater than or equal 98% missing
            values

        drop_unnecessary_cols
            drop specific columns

        assign_col_types
            assign columns types as either string if categorical
            or float is continuous
        """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def preprocess_data(self):
        # generic preprocessing functions
        self.label_missing_values("-9")
        self.label_missing_values(-9)
        self.label_missing_values(pd.NA)
        self.calculate_na_proportions()
        self.remove_all_na_cols()
        self.drop_unessersary_cols()
        self.assign_col_types()

        # categorical variable preprocessing
        self.impute_cat_missing_values()
        self.one_hot_encode_cat_variables()

        # continuous variable preprocessing
        self.normalise_cont_data()
        self.impute_cont_missing_values()

        # input data formatting
        self.concatenate_cat_cont_data()

    def label_missing_values(self, value_to_label):
        """
        Parameters
        ----------
        value_to_label : str, int or float
            value to replace with np.nan
        """
        col_types = self.X_data.dtypes
        self.X_data = self.X_data.replace({value_to_label: np.nan})
        self.X_data = self.X_data.astype(col_types)

    def calculate_na_proportions(self):
        self.col_na_proportion = self.X_data.isnull().sum() / len(self.X_data)

    def remove_all_na_cols(self):
        all_na_cols = self.col_na_proportion[self.col_na_proportion >= 0.98]
        all_na_cols = all_na_cols.index
        self.X_data = self.X_data.drop(all_na_cols, axis=1)

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

    def assign_col_types(self):
        self.cat_data = self.X_data.select_dtypes("string")
        self.cont_data = self.X_data.select_dtypes("float")

    def concatenate_cat_cont_data(self):
        self.X_data = pd.concat([self.cat_data, self.cont_data], axis=1)
        self.features = self.X_data.columns

