import pandas as pd
import seaborn as sns
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb


def clean_df(df: pd.DataFrame) -> pd.DataFrame: ## could make this much more robust to catch more kinds of errors, right not just catches errors in the training dataset
    """
    A function to clean the df by removing NA values from all columns and negative values from the oldpeak columns

    Args:
        df (pd.DataFrame): An unclean data frame

    Returns:
        pd.DataFrame: A clean data frame
        list(int): The list of the indexs of all the dropped cols
    """
    
    og_idx = df.index
    cleaned_df = df.dropna()
    cleaned_df = cleaned_df.loc[cleaned_df['oldpeak'] >= 0, : ]
    cleaned_idx = cleaned_df.index
    dropped_idx = list(set(og_idx) - set(cleaned_idx))
    return cleaned_df.reset_index(drop = True), dropped_idx

def get_preds(inputs: pd.DataFrame, data_loaded: dict) -> pd.DataFrame:
    """
    A func to get the preds and append them to the last col of the df

    Args:
        inputs (pd.DataFrame): The input dataframe w the features
        data_loaded (dict): The dict with the model saved

    Returns:
        pd.DataFrame: The dataframe w the predictions column
    """
    y_pred = data_loaded['model'].predict(inputs)
    inputs['preds'] = y_pred
    return inputs

def load_data(data_path:str = 'data.pkl') -> dict:
    """
    A func that loads the onehot encorders and the trained moded

    Args:
        data_path (str, optional): Path to the data. Defaults to 'data.pkl'.

    Returns:
        dict: A dict with the one hot encoders and model saved as the values
    """

    with open(data_path, 'rb') as file:
        data_loaded = pickle.load(file)
    return data_loaded

class OneHotEncodeDf(): ## should write full doc strings here for all methods
    
    def __init__(self, df_inputs: pd.DataFrame, data_loaded: dict) -> None:
        """
        A class to do one hot encoding on the categorical data

        Args:
            df_inputs (pd.DataFrame): The data frame to be one hot encoded
            data_loaded (dict): The dict which contains the saved one hot encoding configs
        """
        self.df_inputs = df_inputs
        self.data_loaded = data_loaded
        
    def _encode_chest_pain(self) -> None:
        chest_pain_encoded = self.data_loaded['encoded_chest_pain'].transform(self.df_inputs[['chest pain type']])
        encoded_df = pd.DataFrame(chest_pain_encoded, columns = self.data_loaded['encoded_chest_pain'].get_feature_names_out())
        self.df_inputs = pd.concat([self.df_inputs.drop('chest pain type', axis = 1), encoded_df], axis=1)
        
    def _encode_thal(self) -> None:
        encoded_thal = self.data_loaded['encoded_thal'].transform(self.df_inputs[['thal']])
        encoded_df = pd.DataFrame(encoded_thal, columns= self.data_loaded['encoded_thal'].get_feature_names_out())
        self.df_inputs = pd.concat([self.df_inputs.drop('thal', axis = 1), encoded_df], axis=1)
    
    def do_onehot_encoding(self) -> None:
        self._encode_chest_pain()
        self._encode_thal()
        return(self.df_inputs)