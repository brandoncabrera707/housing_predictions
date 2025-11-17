import numpy as np
import pandas as pd
def add_basement_features(df):
  df = df.toCopy()
  df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
  df['sqft_basement_nonzero'] = df['sqft_basement'].where(df['sqft_basement']> 0, np.nan)
  return df


def encode_cities_val_test_set(df, encoder):
  """Uses one hot encoder to transform the validation and test sets

  Args:
      df (DataFrame): DataFrame that is going to be encoded
      encoder (OneHotEncoder): Fitted OneHotEncoder object

  Returns:
      DataFrame: the orignal DataFrame that was passed in, concatenated with the one hot encoded cities DataFrame 
  """

  one_hot_encoded_cities = encoder.transform(df[['city']]) #only transform the city column 
  one_hot_encoded_cities_df = pd.DataFrame(one_hot_encoded_cities, columns = encoder.get_feature_names_out(), index = df.index)
  df = pd.concat([df, one_hot_encoded_cities_df], axis = 1)
  df.drop('city', inplace = True, axis = 1) # No need for city column anymore
  return df

def apply_standard_scaler(df, scaler):
  """Applies standardization to the DataFrame passed into using the StandardScaler object passed into it

  Args:
      df (DataFrame): DataFrame that is going to be standardized
      scaler (StandardScaler): Fitted StandardScaler object we want to use on the dataset

  Returns:
      DataFrame: Returns a scaled version of the DataFrame 
  """

  df_nparray = scaler.transform(df) # only transform data since scaler is already fitted
  df_scaled = pd.DataFrame(df_nparray, index = df.index, columns = df.columns)
  return df_scaled